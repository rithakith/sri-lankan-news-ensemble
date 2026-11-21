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
