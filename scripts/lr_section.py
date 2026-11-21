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
