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
