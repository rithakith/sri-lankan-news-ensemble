"""Inference and evaluation for the notebook's frozen model bundle.
Code generated with assistance from ChatGPT; reviewed and tested for this project.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

MODEL_NAMES = ['LR', 'Random Forest', 'XGBoost', 'Stacking', 'ANN', 'Selected hybrid']
BASE_NAMES = {'LR': 'LR', 'RF': 'Random Forest', 'XGB': 'XGBoost'}
POLICY = 'headline_plus_first_200_body_words'


def model_text(headline, body):
    return headline + '\n' + ' '.join(body.split()[:200])


def validate_bundle(bundle):
    required = {'ann_pipeline', 'stack_base_pipelines', 'stack_meta', 'base_order',
                'label_encoder', 'stack_weight', 'input_policy'}
    if required - bundle.keys():
        raise ValueError(f'Model bundle is missing: {sorted(required - bundle.keys())}')
    if bundle['input_policy'] != POLICY:
        raise ValueError('Unsupported text preprocessing policy.')
    labels = bundle['label_encoder'].classes_
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('Invalid category mapping.')
    expected = np.arange(len(labels))
    order = bundle['base_order']
    if len(order) != 3 or set(order) != set(BASE_NAMES):
        raise ValueError('Expected LR, RF and XGB exactly once in saved base order.')
    for model in [bundle['ann_pipeline'], bundle['stack_meta'],
                  *[bundle['stack_base_pipelines'][key] for key in order]]:
        if not np.array_equal(model.classes_, expected):
            raise ValueError('Class order differs between ensemble components.')
    if not np.isfinite(bundle['stack_weight']) or not 0 <= bundle['stack_weight'] <= 1:
        raise ValueError('Invalid saved hybrid weight.')
    return bundle


def check_probs(probs, rows, classes):
    probs = np.asarray(probs)
    if (probs.shape != (rows, classes) or not np.isfinite(probs).all()
            or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Model returned invalid probabilities.')
    return probs


def infer(bundle, texts):
    validate_bundle(bundle)
    k = len(bundle['label_encoder'].classes_)
    result, timing, blocks = {}, {}, []
    for key in bundle['base_order']:
        start = time.perf_counter()
        probs = check_probs(bundle['stack_base_pipelines'][key].predict_proba(texts), len(texts), k)
        timing[BASE_NAMES[key]] = (time.perf_counter() - start) * 1000
        result[BASE_NAMES[key]] = probs
        blocks.append(probs)
    start = time.perf_counter()
    result['Stacking'] = check_probs(bundle['stack_meta'].predict_proba(np.hstack(blocks)), len(texts), k)
    timing['Stacking'] = sum(timing.values()) + (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    result['ANN'] = check_probs(bundle['ann_pipeline'].predict_proba(texts), len(texts), k)
    timing['ANN'] = (time.perf_counter() - start) * 1000
    result['Selected hybrid'] = blend(result['Stacking'], result['ANN'], bundle['stack_weight'])
    timing['Selected hybrid'] = timing['Stacking'] + timing['ANN']
    return result, timing


def blend(stack, ann, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Weight must be between zero and one.')
    return weight * stack + (1 - weight) * ann


def read_run(root):
    root = Path(root)
    path = root / 'run.json'
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    if run.get('input_policy') != POLICY or not run.get('categories'):
        raise ValueError('Missing categories or unsupported preprocessing in run.json.')
    return run


def predictions(root, split, categories):
    path = Path(root) / f'{split}_predictions.csv'
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={'doc_id': str}, keep_default_na=False)
    required = {'doc_id', 'headline', 'BART_reference', *MODEL_NAMES}
    if required - set(frame):
        raise ValueError(f'{split} predictions lack columns: {sorted(required - set(frame))}')
    if frame.empty or frame.doc_id.duplicated().any():
        raise ValueError(f'{split} predictions must contain unique articles.')
    for col in ['BART_reference', *MODEL_NAMES]:
        if not set(frame[col]) <= set(categories):
            raise ValueError(f'{split}: unexpected categories in {col}.')
    return frame


def scores(frame, categories):
    if frame.empty:
        return pd.DataFrame()
    target = frame.BART_reference
    return pd.DataFrame([{'Model': name,
        'Agreement': accuracy_score(target, frame[name]),
        'Macro-F1': f1_score(target, frame[name], labels=categories, average='macro', zero_division=0),
        'Weighted-F1': f1_score(target, frame[name], average='weighted', zero_division=0)}
        for name in MODEL_NAMES])


def category_scores(frame, model, categories):
    p, r, f, n = precision_recall_fscore_support(frame.BART_reference, frame[model],
                                               labels=categories, zero_division=0)
    # Unsupported categories remain in overall macro-F1 but are visibly unmeasured here.
    return pd.DataFrame({'Category': categories, 'Precision': p, 'Recall': r,
                         'F1': np.where(n > 0, f, np.nan), 'Articles': n})


def matrix(frame, model, categories):
    return confusion_matrix(frame.BART_reference, frame[model], labels=categories)
