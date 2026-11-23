"""Frozen experiment inputs and saved TF-IDF/LinearSVC inference."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODEL = Path(__file__).resolve().parent / 'models/final_classifier.joblib'
STAGES = {'initial': 'Initial', 'baseline_refined': 'Baseline refined',
          'selected_refined': 'Selected refined', 'weighted_refined': 'Weighted refined'}
AUDIT_NAMES = {'BART': 'BART', 'Initial ensemble': 'Stacking · initial',
               'Refined ensemble': 'Stacking · selected refined',
               'Final classifier': 'LinearSVC · selected refined'}

def table(name):
    return pd.read_csv(RESULTS / name)

def selection():
    return json.loads((RESULTS / 'frozen_selection.json').read_text())

def load_classifier(path=MODEL):
    # Only load the trusted repository artifact, never an uploaded pickle.
    saved = joblib.load(path)
    if saved.get('model_name') != 'LinearSVC' or saved.get('stage') != 'selected_refined':
        raise ValueError('Expected the frozen selected-refined LinearSVC artifact.')
    if len(saved['topics']) != len(saved['pipeline'].classes_):
        raise ValueError('Category mapping does not match the classifier.')
    return saved

def predict_article(saved, headline, body):
    if not (headline.strip() or body.strip()):
        raise ValueError('Enter a headline or article body.')
    text = headline.strip() + '\n' + body.strip()
    scores = np.asarray(saved['pipeline'].decision_function([text]))[0]
    classes = np.asarray(saved['pipeline'].classes_, dtype=int)
    order = np.argsort(scores)[::-1]
    alternatives = pd.DataFrame({'Category': [saved['topics'][classes[i]] for i in order[:3]],
                                 'Decision score': scores[order[:3]]})
    vector = saved['pipeline'].named_steps['tfidf'].transform([text])
    return {'category': alternatives.Category.iloc[0], 'margin': float(scores[order[0]]-scores[order[1]]),
            'alternatives': alternatives, 'known_terms': vector.nnz}

def audit_matrix(reference, prediction, topics):
    audit = table('human_audit_joined_predictions.csv')
    return confusion_matrix(audit[reference], audit[prediction], labels=topics, normalize='true')
