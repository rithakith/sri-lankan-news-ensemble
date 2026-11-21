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
