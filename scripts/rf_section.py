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
