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
