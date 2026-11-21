# %% [markdown]
# ## 1. Dataset
# 
# The dataset contains 10,000 English news articles across 13 categories. Reference labels were generated using BART; evaluation measures agreement with these labels rather than independently verified topic accuracy.

# %%
from google.colab import drive
from pathlib import Path
import csv, io, gzip, base64, json, random, re, unicodedata, hashlib
from collections import Counter, defaultdict
csv.field_size_limit(100_000_000)
drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/LankaNews')
candidates = list(DATA_ROOT.rglob('news_english_pseudo_labelled_v8_last10000.csv'))
assert len(candidates) == 1, f'Expected one completed labelled CSV: {candidates}'
INPUT_CSV = str(candidates[0])
OUTPUT_DIR = DATA_ROOT / 'ensemble_training_v8_last10000'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
MODE = 'full_v8'
raw = Path(INPUT_CSV).read_bytes()
reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
original_columns = reader.fieldnames
rows = list(reader)
assert len(rows) == 10000
assert {'doc_id','description','text','predicted_topic','taxonomy_version'} <= set(original_columns)
assert len({r['doc_id'] for r in rows}) == len(rows)
assert all(r['text'].strip() and r['predicted_topic'].strip() for r in rows)
assert {r['taxonomy_version'] for r in rows} == {'v8'}
counts = Counter(r['predicted_topic'] for r in rows)
print('Articles:', len(rows), '| Categories:', len(counts))
print('Input:', INPUT_CSV, '| Output:', OUTPUT_DIR)
print(counts)


# %% [markdown]
# ## 2. Train–validation–test split
# 
# The target split is 70:15:15. Articles with identical normalized headlines or bodies remain in the same split to reduce duplicate leakage.

# %%
# Group exact normalized bodies OR exact normalized headlines together.
# This does NOT detect all semantic near-duplicates; inspect story overlap before final evaluation.
def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()

parent = list(range(len(rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

seen = {}
for i, row in enumerate(rows):
    for field in ('description', 'text'):
        value = normalize(row[field])
        if value:
            key = (field, hashlib.sha256(value.encode()).hexdigest())
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
groups = defaultdict(list)
for i in range(len(rows)):
    groups[find(i)].append(i)

# Category stratification is approximate because complete groups stay together.
# Conflicting-label groups and classes with <3 groups stay in training only.
by_topic = defaultdict(list)
train_only = []
for group in groups.values():
    topics = {rows[i]['predicted_topic'] for i in group}
    if len(topics) > 1:
        train_only.append(group)
    else:
        by_topic[next(iter(topics))].append(group)

rng = random.Random(SEED)
assigned = {name: [] for name in ('train', 'validation', 'test')}
rare_topics = []
for topic in sorted(by_topic):
    topic_groups = by_topic[topic]
    rng.shuffle(topic_groups)
    n = len(topic_groups)
    if n < 3:
        rare_topics.append(topic)
        train_only.extend(topic_groups)
        continue
    n_test = max(1, round(n * 0.15))
    n_valid = max(1, round(n * 0.15))
    for name, selected in (
        ('test', topic_groups[:n_test]),
        ('validation', topic_groups[n_test:n_test+n_valid]),
        ('train', topic_groups[n_test+n_valid:]),
    ):
        assigned[name].extend(i for group in selected for i in group)
assigned['train'].extend(i for group in train_only for i in group)

all_indices = [i for indices in assigned.values() for i in indices]
assert len(all_indices) == len(rows) == len(set(all_indices))
owner = {i: name for name, indices in assigned.items() for i in indices}
assert all(len({owner[i] for i in group}) == 1 for group in groups.values())
assert all(rows[i]['predicted_topic'] in {rows[j]['predicted_topic'] for j in assigned['train']}
           for name in ('validation', 'test') for i in assigned[name])

print('Category                          train  validation  test')
for topic in sorted(counts):
    values = [sum(rows[i]['predicted_topic'] == topic for i in assigned[name])
              for name in ('train', 'validation', 'test')]
    print(f'{topic:35s} {values[0]:5d} {values[1]:10d} {values[2]:5d}')
print('Totals:', {name: len(indices) for name, indices in assigned.items()})
print('Train-only rare categories:', rare_topics)
print('Grouped splits passed: disjoint IDs, complete coverage, no exact-group leakage.')


# %% [markdown]
# ## 3. Reproducibility
# 
# The random seed, dataset hash, split counts, and grouping rules are recorded. Model inputs consist of the headline and first 200 body words.

# %%
# Preserve every input attribute. Keep metadata out of the later feature matrix.
for name, indices in assigned.items():
    with (OUTPUT_DIR / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=original_columns)
        writer.writeheader()
        writer.writerows(rows[i] for i in sorted(indices))
manifest = {
    'mode': MODE, 'seed': SEED, 'input_sha256': hashlib.sha256(raw).hexdigest(),
    'requested_ratios': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
    'actual_counts': {name: len(indices) for name, indices in assigned.items()},
    'grouping': 'normalized exact headline OR exact body; no semantic duplicate detection',
    'rare_train_only': rare_topics,
    'labels': 'BART pseudo-labels, not independently verified ground truth',
    'model_plan': 'TF-IDF LR, RF, XGBoost, grouped stacking, ANN, and validation-selected stack/ANN blend',
    'limitations': 'Recent chronological subset with model-generated labels; no independently verified ground truth.'
}
(OUTPUT_DIR / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))

# Later model inputs: ONLY headline + first 200 body words, matching teacher input coverage.
def model_text(row):
    return row['description'] + '\n' + ' '.join(row['text'].split()[:200])
train_text = [model_text(rows[i]) for i in assigned['train']]
y_train = [rows[i]['predicted_topic'] for i in assigned['train']]


# %% [markdown]
# ## 4. TF-IDF feature extraction
# 
# Vocabulary and inverse document frequencies are fitted on training data only.

# %%
# 4. TF-IDF.
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# Run the preceding split cells first. Only article text is used as input.
import numpy as np
import pandas as pd
import joblib
import sklearn
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

text_by_split = {
    name: [model_text(rows[i]) for i in indices]
    for name, indices in assigned.items()
}
labels_by_split = {
    name: [rows[i]['predicted_topic'] for i in indices]
    for name, indices in assigned.items()
}
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),  # Single words and two-word phrases.
    min_df=1,            # Retain terms appearing in at least one training article.
    max_features=20000,  # An initial cap, not a tuned optimum.
    sublinear_tf=True,   # Dampen repeated words with 1 + log(count).
    norm='l2',
    dtype=np.float32,
)
# Learn vocabulary and IDF weights from TRAINING articles only.
X_train = vectorizer.fit_transform(text_by_split['train'])
vocabulary_before = vectorizer.vocabulary_.copy()
idf_before = vectorizer.idf_.copy()
X_valid = vectorizer.transform(text_by_split['validation'])
X_test = vectorizer.transform(text_by_split['test'])
y_train = np.array(labels_by_split['train'])
y_valid = np.array(labels_by_split['validation'])
y_test = np.array(labels_by_split['test'])
features = vectorizer.get_feature_names_out()

assert vectorizer.vocabulary_ == vocabulary_before
assert np.array_equal(vectorizer.idf_, idf_before)
assert X_train.shape[1] == X_valid.shape[1] == X_test.shape[1]
for name, matrix, labels in (
    ('train', X_train, y_train), ('validation', X_valid, y_valid), ('test', X_test, y_test)
):
    assert matrix.shape[0] == len(labels) == len(assigned[name])
    assert np.isfinite(matrix.data).all()
    zero_rows = int((matrix.getnnz(axis=1) == 0).sum())
    print(f'{name:10s}: {matrix.shape[0]} articles x {matrix.shape[1]} features; zero vectors: {zero_rows}')
print('Vocabulary/IDF unchanged by validation and test transformations: passed')

# Inspect only a few training articles, without densifying the whole matrix.
example = X_train.getrow(0)
top_positions = np.argsort(example.data)[::-1][:10]
top_columns = example.indices[top_positions]
print('\nExample training headline:', rows[assigned['train'][0]]['description'])
display(pd.DataFrame({'term': features[top_columns], 'TF-IDF weight': example.data[top_positions]}))
print('Small matrix preview: first 3 training articles, example article\'s top terms')
display(pd.DataFrame(
    X_train[:3, top_columns].toarray(), columns=features[top_columns],
    index=[rows[i]['doc_id'] for i in assigned['train'][:3]],
).round(3))

# Save sparse features, matching row IDs/labels, and fitted preprocessing.
FEATURE_DIR = OUTPUT_DIR / 'tfidf'
FEATURE_DIR.mkdir(exist_ok=True)
for name, matrix in [('train', X_train), ('validation', X_valid), ('test', X_test)]:
    save_npz(FEATURE_DIR / f'X_{name}.npz', matrix)
    pd.DataFrame({
        'doc_id': [rows[i]['doc_id'] for i in assigned[name]],
        'target': labels_by_split[name],
    }).to_csv(FEATURE_DIR / f'y_{name}.csv', index=False)
joblib.dump(vectorizer, FEATURE_DIR / 'tfidf_vectorizer.joblib')
settings = {
    'mode': MODE, 'sklearn_version': sklearn.__version__,
    'input_sha256': hashlib.sha256(raw).hexdigest(), 'seed': SEED,
    'input_policy': 'headline_plus_first_200_body_words',
    'fit_split': 'train', 'ngram_range': [1, 2], 'min_df': 1,
    'max_features': 20000, 'sublinear_tf': True, 'norm': 'l2',
    'stop_words': None, 'dtype': 'float32',
    'feature_count': len(features),
}
(FEATURE_DIR / 'settings.json').write_text(json.dumps(settings, indent=2))
print('For stacking, refit TF-IDF inside each training fold using a Pipeline; do not reuse this globally fitted matrix for OOF training.')


# %% [markdown]
# ## 5. Logistic Regression

# %%
# 5. Logistic Regression baseline.
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
import time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

# Provisional settings; no validation-based tuning has happened yet.
lr_model = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
started = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', ConvergenceWarning)
    lr_model.fit(X_train, y_train)
assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'Increase max_iter and rerun'
train_seconds = time.perf_counter() - started
train_predictions = lr_model.predict(X_train)
started = time.perf_counter()
valid_probabilities = lr_model.predict_proba(X_valid)
valid_predictions = lr_model.classes_[valid_probabilities.argmax(axis=1)]
prediction_seconds = time.perf_counter() - started
assert valid_probabilities.shape == (len(y_valid), len(lr_model.classes_))
assert np.isfinite(valid_probabilities).all()
assert np.allclose(valid_probabilities.sum(axis=1), 1.0)

# Report all training categories, including those absent from validation.
all_classes = lr_model.classes_
supported_classes = sorted(set(y_valid))
metrics = {
    'model': 'TF-IDF + Logistic Regression', 'mode': MODE,
    'training_articles': len(y_train), 'validation_articles': len(y_valid),
    'training_agreement': accuracy_score(y_train, train_predictions),
    'validation_agreement': accuracy_score(y_valid, valid_predictions),
    'validation_macro_f1_all_training_classes': f1_score(y_valid, valid_predictions, labels=all_classes, average='macro', zero_division=0),
    'validation_macro_f1_supported_classes': f1_score(y_valid, valid_predictions, labels=supported_classes, average='macro', zero_division=0),
    'validation_weighted_f1': f1_score(y_valid, valid_predictions, average='weighted', zero_division=0),
    'train_seconds': train_seconds, 'validation_prediction_seconds': prediction_seconds,
    'reference': 'BART pseudo-labels; agreement, not independently verified accuracy',
    'test_evaluated': False,
}
print('Validation results against BART pseudo-labels')
display(pd.DataFrame([metrics]).T.rename(columns={0: 'value'}))
report = pd.DataFrame(classification_report(
    y_valid, valid_predictions, labels=all_classes, output_dict=True, zero_division=0,
)).T
display(report.round(3))
print('Categories missing from validation:', sorted(set(all_classes) - set(y_valid)))

validation_results = pd.DataFrame({
    'doc_id': [rows[i]['doc_id'] for i in assigned['validation']],
    'headline': [rows[i]['description'] for i in assigned['validation']],
    'bart_reference': y_valid,
    'lr_prediction': valid_predictions,
    'lr_probability': valid_probabilities.max(axis=1),
})
validation_results['matches_bart'] = validation_results['bart_reference'] == validation_results['lr_prediction']
display(validation_results)

MODEL_DIR = OUTPUT_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)
# Bundle fitted text preprocessing with the model for future raw-text inference.
# This fitted baseline is NOT the OOF model used to train a stack.
lr_pipeline = Pipeline([('tfidf', vectorizer), ('lr', lr_model)])
assert np.array_equal(lr_pipeline.predict(text_by_split['validation']), valid_predictions)
joblib.dump(lr_pipeline, MODEL_DIR / 'tfidf_lr_pipeline.joblib')
validation_results.to_csv(MODEL_DIR / 'lr_validation_predictions.csv', index=False)
report.to_csv(MODEL_DIR / 'lr_validation_report.csv')
(MODEL_DIR / 'lr_metrics.json').write_text(json.dumps(metrics, indent=2))


# %% [markdown]
# ## 6. Random Forest

# %%
# 6. Random Forest bagging baseline.
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

# Each tree sees a bootstrap sample of training rows and random feature subsets.
# Use the same sparse TF-IDF features as LR; no SVD or extra fitting here.
# These are initial settings, not validation-tuned settings.
rf_model = RandomForestClassifier(
    n_estimators=200,
    bootstrap=True,
    max_features='sqrt',
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1,
)
started = time.perf_counter()
rf_model.fit(X_train, y_train)
rf_train_seconds = time.perf_counter() - started
rf_train_predictions = rf_model.predict(X_train)
started = time.perf_counter()
rf_valid_probabilities = rf_model.predict_proba(X_valid)
rf_valid_predictions = rf_model.classes_[rf_valid_probabilities.argmax(axis=1)]
rf_prediction_seconds = time.perf_counter() - started
assert np.array_equal(rf_model.classes_, lr_model.classes_)
assert rf_valid_probabilities.shape == valid_probabilities.shape
assert np.isfinite(rf_valid_probabilities).all()
assert np.allclose(rf_valid_probabilities.sum(axis=1), 1.0)

rf_metrics = {
    'model': 'TF-IDF + Random Forest', 'mode': MODE,
    'training_articles': len(y_train), 'validation_articles': len(y_valid),
    'training_agreement': accuracy_score(y_train, rf_train_predictions),
    'validation_agreement': accuracy_score(y_valid, rf_valid_predictions),
    'validation_macro_f1_all_training_classes': f1_score(y_valid, rf_valid_predictions, labels=all_classes, average='macro', zero_division=0),
    'validation_macro_f1_supported_classes': f1_score(y_valid, rf_valid_predictions, labels=supported_classes, average='macro', zero_division=0),
    'validation_weighted_f1': f1_score(y_valid, rf_valid_predictions, average='weighted', zero_division=0),
    'train_seconds': rf_train_seconds,
    'validation_prediction_seconds': rf_prediction_seconds,
    'reference': 'BART pseudo-labels; agreement, not independently verified accuracy',
    'test_evaluated': False,
}
print('Random Forest: 200 trees, bootstrap=True, random feature subsets')
print('Validation agreement with BART')
comparison = pd.DataFrame([metrics, rf_metrics]).set_index('model')
columns = ['training_agreement', 'validation_agreement',
           'validation_macro_f1_all_training_classes', 'validation_weighted_f1', 'train_seconds']
display(comparison[columns].round(4))
print('Validation predicted-category counts:', dict(Counter(rf_valid_predictions.tolist())))
rf_report = pd.DataFrame(classification_report(
    y_valid, rf_valid_predictions, labels=all_classes, output_dict=True, zero_division=0,
)).T
display(rf_report.round(3))

rf_validation_results = pd.DataFrame({
    'doc_id': [rows[i]['doc_id'] for i in assigned['validation']],
    'headline': [rows[i]['description'] for i in assigned['validation']],
    'bart_reference': y_valid,
    'lr_prediction': valid_predictions,
    'rf_prediction': rf_valid_predictions,
    'rf_probability': rf_valid_probabilities.max(axis=1),
})
rf_validation_results['rf_matches_bart'] = y_valid == rf_valid_predictions
display(rf_validation_results)

rf_pipeline = Pipeline([('tfidf', vectorizer), ('rf', rf_model)])
assert np.array_equal(rf_pipeline.predict(text_by_split['validation']), rf_valid_predictions)
joblib.dump(rf_pipeline, MODEL_DIR / 'tfidf_rf_pipeline.joblib')
rf_validation_results.to_csv(MODEL_DIR / 'rf_validation_predictions.csv', index=False)
rf_report.to_csv(MODEL_DIR / 'rf_validation_report.csv')
(MODEL_DIR / 'rf_metrics.json').write_text(json.dumps(rf_metrics, indent=2))
comparison.to_csv(MODEL_DIR / 'baseline_comparison.csv')
print('Training agreement alone is not evidence of generalisation. Test set remains unused.')


# %% [markdown]
# ## 7. XGBoost

# %%
# 7. XGBoost: gradient boosting.
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# XGBoost needs integer targets 0..K-1; retain the mapping back to topic names.
topic_encoder = LabelEncoder().fit(y_train)
encoded_train = topic_encoder.transform(y_train)
xgb_model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    objective='multi:softprob', num_class=len(topic_encoder.classes_),
    eval_metric='mlogloss', tree_method='hist',
    subsample=1.0, colsample_bytree=1.0,
    reg_lambda=1.0, random_state=SEED, n_jobs=2,
)
# Sparse absent entries are treated as missing by XGBoost, unlike explicit zero
# values. Keep sparse input consistent in training, validation and deployment.
started = time.perf_counter()
xgb_model.fit(X_train, encoded_train)
xgb_train_seconds = time.perf_counter() - started
xgb_train_predictions = topic_encoder.inverse_transform(xgb_model.predict(X_train).astype(int))
started = time.perf_counter()
xgb_valid_probabilities = xgb_model.predict_proba(X_valid)
xgb_valid_predictions = topic_encoder.inverse_transform(xgb_valid_probabilities.argmax(axis=1))
xgb_prediction_seconds = time.perf_counter() - started
assert np.array_equal(topic_encoder.classes_, all_classes)
assert xgb_valid_probabilities.shape == valid_probabilities.shape
assert np.isfinite(xgb_valid_probabilities).all()
assert np.allclose(xgb_valid_probabilities.sum(axis=1), 1.0, atol=1e-5)
xgb_metrics = {
    'model': 'TF-IDF + XGBoost', 'mode': MODE,
    'training_articles': len(y_train), 'validation_articles': len(y_valid),
    'training_agreement': accuracy_score(y_train, xgb_train_predictions),
    'validation_agreement': accuracy_score(y_valid, xgb_valid_predictions),
    'validation_macro_f1_all_training_classes': f1_score(y_valid, xgb_valid_predictions, labels=all_classes, average='macro', zero_division=0),
    'validation_macro_f1_supported_classes': f1_score(y_valid, xgb_valid_predictions, labels=supported_classes, average='macro', zero_division=0),
    'validation_weighted_f1': f1_score(y_valid, xgb_valid_predictions, average='weighted', zero_division=0),
    'train_seconds': xgb_train_seconds,
    'validation_prediction_seconds': xgb_prediction_seconds,
    'reference': 'BART pseudo-labels; agreement, not independently verified accuracy',
    'test_evaluated': False,
}
print('XGBoost: 100 boosting rounds, depth 3, learning rate 0.1; initial untuned settings')
print('Validation agreement with BART')
comparison = pd.DataFrame([metrics, rf_metrics, xgb_metrics]).set_index('model')
display(comparison[columns].round(4))
print('Validation predicted-category counts:', dict(Counter(xgb_valid_predictions.tolist())))
xgb_report = pd.DataFrame(classification_report(
    y_valid, xgb_valid_predictions, labels=all_classes, output_dict=True, zero_division=0,
)).T
display(xgb_report.round(3))
xgb_results = pd.DataFrame({
    'doc_id': [rows[i]['doc_id'] for i in assigned['validation']],
    'headline': [rows[i]['description'] for i in assigned['validation']],
    'bart_reference': y_valid, 'lr_prediction': valid_predictions,
    'rf_prediction': rf_valid_predictions, 'xgb_prediction': xgb_valid_predictions,
    'xgb_probability': xgb_valid_probabilities.max(axis=1),
})
display(xgb_results)
# Export separately: vectorizer transforms raw text; encoder maps numeric outputs.
joblib.dump({'vectorizer': vectorizer, 'model': xgb_model, 'label_encoder': topic_encoder},
            MODEL_DIR / 'tfidf_xgb_bundle.joblib')
xgb_model.save_model(MODEL_DIR / 'xgb_model.json')
xgb_results.to_csv(MODEL_DIR / 'xgb_validation_predictions.csv', index=False)
xgb_report.to_csv(MODEL_DIR / 'xgb_validation_report.csv')
(MODEL_DIR / 'xgb_metrics.json').write_text(json.dumps(xgb_metrics, indent=2))
comparison.to_csv(MODEL_DIR / 'baseline_comparison.csv')


# %% [markdown]
# ## 8. Stacking ensemble
# 
# The Logistic Regression meta-model is trained on out-of-fold probabilities from the three base models. TF-IDF is refitted within each grouped fold.

# %%
# 8. Out-of-fold stacking.
from sklearn.base import clone

train_indices = np.array(assigned['train'])
group_counts = defaultdict(set)
for i in train_indices:
    group_counts[rows[i]['predicted_topic']].add(find(int(i)))
excluded = sorted(topic for topic, ids in group_counts.items() if len(ids) < 2)
assert not excluded, f'Insufficient training groups: {excluded}'
eligible = sorted(set(group_counts) - set(excluded))
stack_train_ids = np.array([i for i in train_indices if rows[i]['predicted_topic'] in eligible])
stack_valid_ids = np.array([i for i in assigned['validation'] if rows[i]['predicted_topic'] in eligible])
stack_text = np.array([model_text(rows[i]) for i in stack_train_ids])
stack_valid_text = [model_text(rows[i]) for i in stack_valid_ids]
stack_encoder = LabelEncoder().fit(eligible)
stack_y = stack_encoder.transform([rows[i]['predicted_topic'] for i in stack_train_ids])
stack_valid_y = stack_encoder.transform([rows[i]['predicted_topic'] for i in stack_valid_ids])
stack_groups = np.array([find(int(i)) for i in stack_train_ids])
n_folds = min(5, min(len(group_counts[t]) for t in eligible))
# Keep complete story groups together even when their BART labels differ.
from sklearn.model_selection import StratifiedGroupKFold
K = len(eligible)
fold_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
folds = list(fold_splitter.split(stack_text, stack_y, groups=stack_groups))
for fit_ids, holdout_ids in folds:
    assert set(stack_groups[fit_ids]).isdisjoint(stack_groups[holdout_ids])
    assert set(stack_y[fit_ids]) == set(range(K)), 'Fold lacks a training class; revise folds or enlarge sample'
print('Stacking experiment categories:', eligible)
print('Excluded categories:', excluded)
print(f'Training articles: {len(stack_y)}; validation articles: {len(stack_valid_y)}; folds: {n_folds}')

# Every base model owns its preprocessing, which is refitted inside each fold.
base_models = {
    'LR': Pipeline([('tfidf', clone(vectorizer)), ('model', clone(lr_model))]),
    'RF': Pipeline([('tfidf', clone(vectorizer)), ('model', clone(rf_model))]),
    'XGB': Pipeline([('tfidf', clone(vectorizer)), ('model', clone(xgb_model).set_params(num_class=K))]),
}
oof = np.full((len(stack_y), len(base_models) * K), np.nan)
fold_number = np.full(len(stack_y), -1)
started = time.perf_counter()
for fold_id, (fit_ids, holdout_ids) in enumerate(folds):
    fold_number[holdout_ids] = fold_id
    for model_id, (name, template) in enumerate(base_models.items()):
        model = clone(template)
        model.fit(stack_text[fit_ids].tolist(), stack_y[fit_ids])
        assert np.array_equal(model.classes_, np.arange(K))
        probs = model.predict_proba(stack_text[holdout_ids].tolist())
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
        oof[holdout_ids, model_id*K:(model_id+1)*K] = probs
    print(f'Completed OOF fold {fold_id+1}/{n_folds}')
assert (fold_number >= 0).all() and np.isfinite(oof).all()
# Meta-model sees ONLY OOF probabilities and training labels.
stack_meta = LogisticRegression(C=1.0, max_iter=2000)
stack_meta.fit(oof, stack_y)

# Refit base pipelines on all eligible training articles for new-article inference.
fitted_bases = {}
valid_blocks = []
same_subset_results = []
for name, template in base_models.items():
    model = clone(template)
    model.fit(stack_text.tolist(), stack_y)
    probs = model.predict_proba(stack_valid_text)
    valid_blocks.append(probs)
    fitted_bases[name] = model
    pred = probs.argmax(axis=1)
    same_subset_results.append({'model': name, 'validation_agreement': accuracy_score(stack_valid_y, pred),
        'macro_f1_all_experiment_classes': f1_score(stack_valid_y, pred, labels=np.arange(K), average='macro', zero_division=0)})
stack_valid_features = np.hstack(valid_blocks)
stack_valid_probs = stack_meta.predict_proba(stack_valid_features)
stack_valid_pred = stack_valid_probs.argmax(axis=1)
assert np.array_equal(stack_meta.classes_, np.arange(K))
assert np.allclose(stack_valid_probs.sum(axis=1), 1)
same_subset_results.append({'model': 'Stack: LR + RF + XGB → LR',
    'validation_agreement': accuracy_score(stack_valid_y, stack_valid_pred),
    'macro_f1_all_experiment_classes': f1_score(stack_valid_y, stack_valid_pred, labels=np.arange(K), average='macro', zero_division=0)})
stack_comparison = pd.DataFrame(same_subset_results).set_index('model')
display(stack_comparison.round(4))
print('OOF matrix shape:', oof.shape, '= articles x (3 models x categories)')
print('Total stacking experiment time:', round(time.perf_counter() - started, 2), 'seconds')
print('Stack predicted categories:', dict(Counter(stack_encoder.inverse_transform(stack_valid_pred))))

STACK_DIR = OUTPUT_DIR / 'stacking'
STACK_DIR.mkdir(exist_ok=True)
feature_names = [f'{name}__{topic}' for name in base_models for topic in eligible]
oof_table = pd.DataFrame(oof, columns=feature_names)
oof_table.insert(0, 'doc_id', [rows[i]['doc_id'] for i in stack_train_ids])
oof_table.insert(1, 'fold', fold_number)
oof_table.insert(2, 'bart_reference', stack_encoder.inverse_transform(stack_y))
oof_table.to_csv(STACK_DIR / 'oof_training_predictions.csv', index=False)
stack_comparison.to_csv(STACK_DIR / 'same_subset_comparison.csv')
pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in stack_valid_ids],
    'bart_reference': stack_encoder.inverse_transform(stack_valid_y),
    'stack_prediction': stack_encoder.inverse_transform(stack_valid_pred),
}).to_csv(STACK_DIR / 'validation_predictions.csv', index=False)
joblib.dump({'base_pipelines': fitted_bases, 'meta_model': stack_meta,
    'label_encoder': stack_encoder, 'model_order': list(base_models),
    'feature_names': feature_names, 'input_policy': 'headline_plus_first_200_body_words'},
    STACK_DIR / 'stack_bundle.joblib')
(STACK_DIR / 'settings.json').write_text(json.dumps({'mode': MODE, 'folds': n_folds,
    'excluded_categories': excluded, 'categories': eligible, 'seed': SEED,
    'train_articles': len(stack_y), 'validation_articles': len(stack_valid_y),
    'grouping': 'normalized exact headline/body', 'preprocessing': 'refitted inside every OOF fold',
    'reference': 'BART pseudo-labels; validation agreement', 'test_evaluated': False}, indent=2))


# %% [markdown]
# ## 9. Artificial neural network and hybrid ensemble
# 
# Validation loss selects the ANN epoch. Validation macro-F1 selects the stack/ANN probability blend weight.

# %%
# 9. ANN and stacking + ANN hybrid.
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from copy import deepcopy

# Match the stack's experiment subset and class order exactly.
# Refit preprocessing on eligible TRAINING articles, never on validation/test.
ann_vectorizer = clone(vectorizer)
ann_sparse_train = ann_vectorizer.fit_transform(stack_text.tolist())
ann_sparse_valid = ann_vectorizer.transform(stack_valid_text)
n_components = min(100, ann_sparse_train.shape[0] - 1, ann_sparse_train.shape[1] - 1)
ann_svd = TruncatedSVD(n_components=n_components, random_state=SEED)
ann_train_reduced = ann_svd.fit_transform(ann_sparse_train)
ann_valid_reduced = ann_svd.transform(ann_sparse_valid)
ann_scaler = StandardScaler()
ann_train_features = ann_scaler.fit_transform(ann_train_reduced)
ann_valid_features = ann_scaler.transform(ann_valid_reduced)

# A single hidden layer is an ANN. CPU is sufficient for this small network.
# Train one epoch at a time; validation log loss selects the stopping point.
from sklearn.metrics import log_loss
ann_model = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
    alpha=0.01, batch_size=min(32, len(stack_y)), learning_rate_init=0.001,
    random_state=SEED, early_stopping=False, max_iter=1)
history = []
best_loss = float('inf')
best_epoch = 0
started = time.perf_counter()
for epoch in range(1, 201):
    ann_model.partial_fit(ann_train_features, stack_y, classes=np.arange(K))
    probs = ann_model.predict_proba(ann_valid_features)
    loss = log_loss(stack_valid_y, probs, labels=np.arange(K))
    history.append({'epoch': epoch, 'training_loss': ann_model.loss_, 'validation_log_loss': loss})
    if loss < best_loss - 0.0001:
        best_loss, best_epoch = loss, epoch
        best_ann = deepcopy(ann_model)
    if epoch - best_epoch >= 20:
        break
ann_model = best_ann
ann_valid_probs = ann_model.predict_proba(ann_valid_features)
assert np.array_equal(ann_model.classes_, stack_meta.classes_)
assert ann_valid_probs.shape == stack_valid_probs.shape
assert np.isfinite(ann_valid_probs).all()
assert np.allclose(ann_valid_probs.sum(axis=1), 1)
ann_pred = ann_valid_probs.argmax(axis=1)
ann_train_agreement = accuracy_score(stack_y, ann_model.predict(ann_train_features))
print(f'ANN: {n_components} SVD inputs → 32 ReLU neurons → {K} category outputs')
print(f'Best epoch: {best_epoch}; epochs run: {epoch}; training agreement: {ann_train_agreement:.3f}')
print('Training time:', round(time.perf_counter() - started, 2), 'seconds')

# alpha=0 means ANN alone; alpha=1 means stack alone.
# A fixed grid includes endpoints so we can see if blending actually helps.
blend_rows = []
for weight in np.linspace(0, 1, 11):
    probs = weight * stack_valid_probs + (1 - weight) * ann_valid_probs
    pred = probs.argmax(axis=1)
    blend_rows.append({'stack_weight': float(weight), 'ann_weight': float(1-weight),
        'validation_agreement': accuracy_score(stack_valid_y, pred),
        'macro_f1': f1_score(stack_valid_y, pred, labels=np.arange(K), average='macro', zero_division=0)})
blend_search = pd.DataFrame(blend_rows)
# First maximum is a deterministic tie-break favouring the ANN endpoint.
best_blend = blend_search.loc[blend_search['macro_f1'].idxmax()]
blend_weight = float(best_blend['stack_weight'])
hybrid_valid_probs = blend_weight * stack_valid_probs + (1-blend_weight) * ann_valid_probs
hybrid_pred = hybrid_valid_probs.argmax(axis=1)
assert np.allclose(hybrid_valid_probs.sum(axis=1), 1)
ann_comparison = stack_comparison.copy()
ann_comparison.loc['ANN (TF-IDF + SVD)'] = [accuracy_score(stack_valid_y, ann_pred),
    f1_score(stack_valid_y, ann_pred, labels=np.arange(K), average='macro', zero_division=0)]
ann_comparison.loc['Stack + ANN (selected blend)'] = [accuracy_score(stack_valid_y, hybrid_pred), float(best_blend['macro_f1'])]
print('SAME SUBSET: validation results against BART pseudo-labels')
display(ann_comparison.round(4))
display(blend_search.round(4))
print(f'Selected: {blend_weight:.0%} stack + {1-blend_weight:.0%} ANN')
if blend_weight in (0.0, 1.0):
    print('Validation selected a single-model endpoint; no hybrid benefit demonstrated.')
else:
    endpoint_best = max(blend_search.iloc[0]['macro_f1'], blend_search.iloc[-1]['macro_f1'])
    print('Improvement over better endpoint:', round(float(best_blend['macro_f1']) - endpoint_best, 4))
print('Validation selected both ANN epoch and blend weight; these are development scores, not an independent performance estimate.')
print('Validation selects the ANN epoch and blend weight. Test data remains unused.')

HYBRID_DIR = OUTPUT_DIR / 'ann_hybrid'
HYBRID_DIR.mkdir(exist_ok=True)
ann_pipeline = Pipeline([('tfidf', ann_vectorizer), ('svd', ann_svd),
                         ('scale', ann_scaler), ('ann', ann_model)])
assert np.allclose(ann_pipeline.predict_proba(stack_valid_text), ann_valid_probs)
joblib.dump({'ann_pipeline': ann_pipeline, 'stack_base_pipelines': fitted_bases,
    'stack_meta': stack_meta, 'base_order': list(base_models),
    'label_encoder': stack_encoder, 'stack_weight': blend_weight,
    'input_policy': 'headline_plus_first_200_body_words'}, HYBRID_DIR / 'hybrid_bundle.joblib')
pd.DataFrame(history).to_csv(HYBRID_DIR / 'ann_training_history.csv', index=False)
blend_search.to_csv(HYBRID_DIR / 'blend_validation_search.csv', index=False)
ann_comparison.to_csv(HYBRID_DIR / 'same_subset_comparison.csv')
pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in stack_valid_ids],
    'bart_reference': stack_encoder.inverse_transform(stack_valid_y),
    'ann_prediction': stack_encoder.inverse_transform(ann_pred),
    'hybrid_prediction': stack_encoder.inverse_transform(hybrid_pred),
}).to_csv(HYBRID_DIR / 'validation_predictions.csv', index=False)
(HYBRID_DIR / 'settings.json').write_text(json.dumps({'mode': MODE,
    'categories': eligible, 'excluded_categories': excluded,
    'svd_components': n_components, 'hidden_layer_sizes': [32],
    'alpha_regularisation': 0.01, 'seed': SEED, 'best_epoch': best_epoch,
    'stack_weight': blend_weight, 'test_evaluated': False,
    'validation_used_for': ['ANN early stopping', 'blend weight selection']}, indent=2))


# %% [markdown]
# ## 10. Validation results
# 
# All models are compared on the same validation articles and categories.

# %%
# 10. Visual comparison and interpretation.
# All model comparisons below use the SAME stacking-experiment validation subset.
# https://scikit-learn.org/stable/modules/model_evaluation.html
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

VIS_DIR = OUTPUT_DIR / 'visual_analysis'
VIS_DIR.mkdir(exist_ok=True)
names = ['LR', 'Random Forest', 'XGBoost', 'Stacking', 'ANN', 'Selected hybrid']
predictions = {
    'LR': fitted_bases['LR'].predict(stack_valid_text),
    'Random Forest': fitted_bases['RF'].predict(stack_valid_text),
    'XGBoost': fitted_bases['XGB'].predict(stack_valid_text),
    'Stacking': stack_valid_pred, 'ANN': ann_pred, 'Selected hybrid': hybrid_pred,
}
assert all(len(p) == len(stack_valid_y) for p in predictions.values())
topic_names = stack_encoder.classes_.tolist()
print(f'Comparison scope: {len(stack_y)} training articles, {len(stack_valid_y)} validation articles, {K} categories.')
print('Excluded from this experiment:', excluded)
print('BART reference labels. Validation results are used for model selection.')

metric_rows = []
class_f1 = []
for name in names:
    pred = predictions[name]
    metric_rows.append({'model': name,
        'agreement': accuracy_score(stack_valid_y, pred),
        'macro_f1': f1_score(stack_valid_y, pred, labels=np.arange(K), average='macro', zero_division=0),
        'weighted_f1': f1_score(stack_valid_y, pred, average='weighted', zero_division=0)})
    class_f1.append(precision_recall_fscore_support(stack_valid_y, pred, labels=np.arange(K), zero_division=0)[2])
metric_table = pd.DataFrame(metric_rows).set_index('model')

# 1: Same-subset scores and training/validation category support.
fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
metric_table.plot.bar(ax=axes[0], ylim=(0, 1), rot=25)
axes[0].set(title='Validation agreement and F1 (same articles)', ylabel='Score (0–1)', xlabel='Model')
support = pd.DataFrame({'training': np.bincount(stack_y, minlength=K),
    'validation': np.bincount(stack_valid_y, minlength=K)}, index=topic_names)
support.plot.barh(ax=axes[1])
axes[1].set(title='Category support: imbalance and missing coverage', xlabel='Articles', ylabel='Category')
fig.savefig(VIS_DIR / '01_scores_and_support.png', dpi=160, bbox_inches='tight')
plt.show()

# 2: Per-category F1. Grey cells mean no reference examples, not measured failure.
f1_matrix = np.array(class_f1)
validation_support = np.bincount(stack_valid_y, minlength=K)
masked_f1 = np.ma.masked_where(np.broadcast_to(validation_support == 0, f1_matrix.shape), f1_matrix)
fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('lightgray')
im = ax.imshow(masked_f1, vmin=0, vmax=1, cmap=cmap, aspect='auto')
ax.set_xticks(np.arange(K), [f'{topic}\n(n={n})' for topic, n in zip(topic_names, validation_support)], rotation=35, ha='right')
ax.set_yticks(np.arange(len(names)), names)
for r in range(len(names)):
    for c in range(K):
        label = 'N/A' if validation_support[c] == 0 else f'{f1_matrix[r,c]:.2f}'
        ax.text(c, r, label, ha='center', va='center', color='white' if f1_matrix[r,c] > 0.6 and validation_support[c] else 'black')
ax.set(title='F1 by category — n is validation support', xlabel='BART category', ylabel='Model')
fig.colorbar(im, ax=ax, label='F1 (0–1)')
fig.savefig(VIS_DIR / '02_category_f1.png', dpi=160, bbox_inches='tight')
plt.show()

# 3: Confusion matrices with identical axes/scales across all models.
fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
short_topics = [t.replace('_', '\n') for t in topic_names]
for ax, name in zip(axes.flat, names):
    cm = confusion_matrix(stack_valid_y, predictions[name], labels=np.arange(K))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(validation_support))
    ax.set_xticks(np.arange(K), short_topics, rotation=90)
    ax.set_yticks(np.arange(K), short_topics)
    ax.tick_params(labelsize=8)
    ax.set(title=name, xlabel='Predicted category', ylabel='BART reference category')
    for r, c in zip(*np.nonzero(cm)):
        ax.text(c, r, str(cm[r,c]), ha='center', va='center', color='white' if cm[r,c] > max(validation_support)/2 else 'black')
fig.colorbar(im, ax=list(axes.flat), label='Articles')
fig.suptitle('Where predictions go wrong: diagonal entries match BART')
fig.savefig(VIS_DIR / '03_confusion_matrices.png', dpi=160, bbox_inches='tight')
plt.show()

# 4: ANN optimisation and hybrid weight sensitivity.
hist = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
axes[0].plot(hist['epoch'], hist['training_loss'], label='Training objective (includes regularisation)')
axes[0].plot(hist['epoch'], hist['validation_log_loss'], label='Validation log loss')
axes[0].axvline(best_epoch, color='gray', linestyle='--', label=f'Selected epoch {best_epoch}')
axes[0].set(title='ANN learning history', xlabel='Epoch', ylabel='Loss (lower is better)')
axes[0].legend(fontsize=8)
axes[1].plot(blend_search['stack_weight'], blend_search['macro_f1'], marker='o', label='Macro-F1')
axes[1].plot(blend_search['stack_weight'], blend_search['validation_agreement'], marker='s', label='Agreement')
axes[1].axvline(blend_weight, color='gray', linestyle='--', label='Selected weight')
axes[1].set(title='Does adding the stack help the ANN?', xlabel='Stack weight (0 = ANN alone; 1 = stack alone)', ylabel='Validation score', ylim=(0, 1))
axes[1].legend(fontsize=8)
fig.savefig(VIS_DIR / '04_ann_learning_and_blending.png', dpi=160, bbox_inches='tight')
plt.show()

# 5: Diversity is useful only if models make complementary correct predictions.
disagreement = np.array([[np.mean(predictions[a] != predictions[b]) for b in names] for a in names])
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(disagreement, cmap='Oranges', vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(len(names)):
        ax.text(c, r, f'{disagreement[r,c]:.0%}', ha='center', va='center')
ax.set(title='Prediction disagreement on the same validation articles')
fig.colorbar(im, ax=ax, label='Fraction of different predictions')
fig.savefig(VIS_DIR / '05_model_disagreement.png', dpi=160, bbox_inches='tight')
plt.show()

stack_correct = stack_valid_pred == stack_valid_y
ann_correct = ann_pred == stack_valid_y
complementarity = pd.Series({
    'Both match BART': int((stack_correct & ann_correct).sum()),
    'Only ANN matches': int((~stack_correct & ann_correct).sum()),
    'Only stack matches': int((stack_correct & ~ann_correct).sum()),
    'Neither matches': int((~stack_correct & ~ann_correct).sum()),
}, name='Articles')
display(complementarity.to_frame())
inspection = pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in stack_valid_ids],
    'headline': [rows[i]['description'] for i in stack_valid_ids],
    'BART_reference': stack_encoder.inverse_transform(stack_valid_y)})
for name in names:
    inspection[name] = stack_encoder.inverse_transform(predictions[name].astype(int))
inspection['models_disagree'] = np.array(list(predictions.values())).max(axis=0) != np.array(list(predictions.values())).min(axis=0)
display(inspection)
inspection.to_csv(VIS_DIR / 'article_level_comparison.csv', index=False)
metric_table.to_csv(VIS_DIR / 'validation_metrics.csv')
support.to_csv(VIS_DIR / 'category_support.csv')
pd.DataFrame(f1_matrix, index=names, columns=topic_names).to_csv(VIS_DIR / 'category_f1.csv')

best_name = metric_table['macro_f1'].idxmax()
print('Evidence-based observations for this run:')
print(f'- Highest validation macro-F1: {best_name} ({metric_table.loc[best_name, "macro_f1"]:.3f}). This is not proof of superiority.')
print(f'- No validation examples for: {[t for t, n in zip(topic_names, validation_support) if n == 0]}.')
print(f'- One changed prediction moves agreement by {1/len(stack_valid_y):.1%}.')
print('- ANN epoch and hybrid weight were selected on this validation set; do not report those scores as independent test performance.')
print('- Inspect the confusion matrices for majority-category collapse; disagreement alone does not imply useful diversity.')
print('- Training/validation loss curves suggest optimisation behaviour, not a causal explanation or proof of generalisation.')


# %% [markdown]
# ## 11. Test results
# 
# The ANN epoch and hybrid weight are fixed before test evaluation.

# %%
# 11. Frozen-model test evaluation and plots.
# No fit(), tuning, epoch selection or blend-weight selection in this section.
test_ids = [i for i in assigned['test'] if rows[i]['predicted_topic'] in eligible]
test_text = [model_text(rows[i]) for i in test_ids]
test_targets = stack_encoder.transform([rows[i]['predicted_topic'] for i in test_ids])
test_predictions = {}
test_base_probs = []
for key, label in [('LR', 'LR'), ('RF', 'Random Forest'), ('XGB', 'XGBoost')]:
    model = fitted_bases[key]
    assert np.array_equal(model.classes_, np.arange(K))
    probs = model.predict_proba(test_text)
    test_base_probs.append(probs)
    test_predictions[label] = probs.argmax(axis=1)
test_stack_probs = stack_meta.predict_proba(np.hstack(test_base_probs))
test_ann_probs = ann_pipeline.predict_proba(test_text)
test_hybrid_probs = blend_weight * test_stack_probs + (1-blend_weight) * test_ann_probs
for probs in [test_stack_probs, test_ann_probs, test_hybrid_probs]:
    assert probs.shape == (len(test_ids), K)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
test_predictions.update({'Stacking': test_stack_probs.argmax(axis=1),
    'ANN': test_ann_probs.argmax(axis=1), 'Selected hybrid': test_hybrid_probs.argmax(axis=1)})
TEST_DIR = OUTPUT_DIR / 'test_analysis'
TEST_DIR.mkdir(exist_ok=True)
print(f'Frozen test evaluation: {len(test_ids)} / {len(assigned["test"])} test articles; {K} experiment categories.')
print('Excluded experiment categories:', excluded)
print(f'Fixed validation-selected hybrid: {blend_weight:.0%} stack + {1-blend_weight:.0%} ANN; fixed ANN epoch {best_epoch}')
print('BART pseudo-label agreement, not independently verified topic accuracy.')

test_metric_rows = []
test_f1 = []
test_reports = {}
for name in names:
    pred = test_predictions[name]
    test_metric_rows.append({'model': name, 'agreement': accuracy_score(test_targets, pred),
        'macro_f1': f1_score(test_targets, pred, labels=np.arange(K), average='macro', zero_division=0),
        'weighted_f1': f1_score(test_targets, pred, average='weighted', zero_division=0)})
    test_f1.append(precision_recall_fscore_support(test_targets, pred, labels=np.arange(K), zero_division=0)[2])
    test_reports[name] = classification_report(test_targets, pred, labels=np.arange(K),
        target_names=topic_names, output_dict=True, zero_division=0)
test_metric_table = pd.DataFrame(test_metric_rows).set_index('model')
display(test_metric_table.round(4))
test_support = np.bincount(test_targets, minlength=K)

fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
test_metric_table.plot.bar(ax=axes[0], rot=20, ylim=(0, 1))
axes[0].set(title=f'Test comparison: same {len(test_ids)} articles, frozen models', xlabel='Model', ylabel='Score (0–1)')
stages = pd.DataFrame({'validation': metric_table['macro_f1'], 'test': test_metric_table['macro_f1']})
stages.plot.bar(ax=axes[1], rot=20, ylim=(0, 1))
axes[1].set(title='Macro-F1: validation versus test', xlabel='Model', ylabel='Macro-F1 (all experiment categories)')
fig.savefig(TEST_DIR / '01_test_scores.png', dpi=160, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
test_f1 = np.array(test_f1)
masked = np.ma.masked_where(np.broadcast_to(test_support == 0, test_f1.shape), test_f1)
cmap = plt.get_cmap('Blues').copy()
cmap.set_bad('lightgray')
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(K), [f'{t}\n(n={n})' for t, n in zip(topic_names, test_support)], rotation=40, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(K):
        ax.text(c, r, 'N/A' if test_support[c] == 0 else f'{test_f1[r,c]:.2f}',
            ha='center', va='center', color='white' if test_f1[r,c] > .6 and test_support[c] else 'black')
ax.set(title='Test F1 by category; n = test support', xlabel='BART reference category', ylabel='Model')
fig.colorbar(im, ax=ax, label='F1')
fig.savefig(TEST_DIR / '02_test_category_f1.png', dpi=160, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
for ax, name in zip(axes.flat, names):
    cm = confusion_matrix(test_targets, test_predictions[name], labels=np.arange(K))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(test_support))
    ax.set_xticks(range(K), short_topics, rotation=90)
    ax.set_yticks(range(K), short_topics)
    ax.tick_params(labelsize=8)
    ax.set(title=name, xlabel='Predicted category', ylabel='BART reference category')
    for r, c in zip(*np.nonzero(cm)):
        ax.text(c, r, str(cm[r,c]), ha='center', va='center',
            color='white' if cm[r,c] > max(test_support)/2 else 'black')
fig.colorbar(im, ax=list(axes.flat), label='Test articles')
fig.suptitle('Test confusion matrices — shared category order and colour scale')
fig.savefig(TEST_DIR / '03_test_confusion.png', dpi=160, bbox_inches='tight')
plt.show()

test_disagreement = np.array([[np.mean(test_predictions[a] != test_predictions[b]) for b in names] for a in names])
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
im = ax.imshow(test_disagreement, cmap='Oranges', vmin=0, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
ax.set_yticks(range(len(names)), names)
for r in range(len(names)):
    for c in range(len(names)):
        ax.text(c, r, f'{test_disagreement[r,c]:.0%}', ha='center', va='center')
ax.set(title='Test prediction disagreement')
fig.colorbar(im, ax=ax, label='Fraction of different predictions')
fig.savefig(TEST_DIR / '04_test_disagreement.png', dpi=160, bbox_inches='tight')
plt.show()

test_inspection = pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in test_ids],
    'headline': [rows[i]['description'] for i in test_ids],
    'BART_reference': stack_encoder.inverse_transform(test_targets)})
for name in names:
    test_inspection[name] = stack_encoder.inverse_transform(test_predictions[name].astype(int))
display(test_inspection)
test_metric_table.to_csv(TEST_DIR / 'test_metrics.csv')
test_inspection.to_csv(TEST_DIR / 'test_article_predictions.csv', index=False)
stages.to_csv(TEST_DIR / 'validation_test_comparison.csv')
(TEST_DIR / 'classification_reports.json').write_text(json.dumps(test_reports, indent=2))
(TEST_DIR / 'evaluation_settings.json').write_text(json.dumps({'mode': MODE,
    'test_articles': len(test_ids), 'original_test_articles': len(assigned['test']),
    'categories': topic_names, 'excluded_categories': excluded,
    'stack_weight_fixed_from_validation': blend_weight, 'ann_epoch_fixed_from_validation': best_epoch,
    'no_test_tuning': True, 'reference': 'BART pseudo-labels',
    'warning': 'Labels are model-generated; no independent human-labelled benchmark is available.'}, indent=2))
print('One changed test prediction changes agreement by', f'{1/len(test_ids):.1%}')
print('Do not tune settings against these test results and continue calling them untouched test results.')
print('No hybrid improvement can be claimed if the frozen selected weight is an endpoint.')

