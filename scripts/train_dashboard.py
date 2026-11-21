"""Run existing notebook model sections locally and export a dashboard experiment.
Generated with ChatGPT assistance. No retuning on the test set.
"""
import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mode', choices=['pilot_v7', 'full_v8'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.input.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    original_columns = reader.fieldnames
    rows = list(reader)
    if not rows or not {'doc_id','description','text','predicted_topic'} <= set(original_columns):
        raise ValueError('Expected labelled news articles with doc_id, description, text and predicted_topic.')
    if args.mode == 'full_v8' and {r.get('taxonomy_version') for r in rows} != {'v8'}:
        raise ValueError('full_v8 requires explicit v8 taxonomy_version on every row.')
    if len({r['doc_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate article IDs.')
    context = dict(globals(), rows=rows, raw=raw, original_columns=original_columns,
                   OUTPUT_DIR=args.output, MODE=args.mode, SEED=42, counts=Counter(r['predicted_topic'] for r in rows),
                   display=lambda value: None)
    nb = json.loads((ROOT/'notebooks/LankaNews_Ensemble_Training.ipynb').read_text())
    # Reuse grouped split and input preparation directly from the notebook.
    for index in [3,5]:
        source=''.join(nb['cells'][index]['source'])
        exec(compile(source,f'notebook-cell-{index}','exec'), context)
    for filename in ['tfidf_section.py','lr_section.py','rf_section.py','xgb_section.py',
                     'stack_section.py','ann_hybrid_section.py']:
        source=(ROOT/'scripts'/filename).read_text()
        if filename == 'stack_section.py':
            source=source.replace('for category in range(K):', 'for category in range(len(eligible)):')
            if args.mode == 'pilot_v7':
                source=source.replace("assert not excluded, f'Insufficient training groups: {excluded}'",
                                      "assert MODE == 'pilot_v7' or not excluded")
                start=source.index('# Keep complete story groups')
                end=source.index('for fit_ids, holdout_ids in folds:',start)
                source=source[:start]+'''# Pilot-only class-balanced complete-group folds.
K = len(eligible)
fold_assignment = np.full(len(stack_y), -1)
fold_rng = random.Random(SEED)
for category in range(K):
    category_groups = sorted(set(stack_groups[stack_y == category]))
    fold_rng.shuffle(category_groups)
    for rank, group_id in enumerate(category_groups):
        mask = stack_groups == group_id
        assert set(stack_y[mask]) == {category}
        fold_assignment[mask] = rank % n_folds
folds = [(np.flatnonzero(fold_assignment != f), np.flatnonzero(fold_assignment == f)) for f in range(n_folds)]
'''+source[end:]
        print('Running',filename,flush=True)
        exec(compile(source,filename,'exec'),context)
    exec('''
from sklearn.metrics import precision_recall_fscore_support
names = ['LR','Random Forest','XGBoost','Stacking','ANN','Selected hybrid']
predictions = {'LR': fitted_bases['LR'].predict(stack_valid_text),
 'Random Forest': fitted_bases['RF'].predict(stack_valid_text),
 'XGBoost': fitted_bases['XGB'].predict(stack_valid_text),
 'Stacking':stack_valid_pred, 'ANN':ann_pred,'Selected hybrid':hybrid_pred}
test_ids = [i for i in assigned['test'] if rows[i]['predicted_topic'] in eligible]
test_text = [model_text(rows[i]) for i in test_ids]
test_predictions = {}
blocks=[]
for key,name in [('LR','LR'),('RF','Random Forest'),('XGB','XGBoost')]:
 probs=fitted_bases[key].predict_proba(test_text)
 blocks.append(probs)
 test_predictions[name]=probs.argmax(axis=1)
stack_probs=stack_meta.predict_proba(np.hstack(blocks))
ann_probs=ann_pipeline.predict_proba(test_text)
test_predictions.update({'Stacking':stack_probs.argmax(axis=1), 'ANN':ann_probs.argmax(axis=1),
 'Selected hybrid':(blend_weight*stack_probs+(1-blend_weight)*ann_probs).argmax(axis=1)})
''',context)
    export=(ROOT/'scripts/export_dashboard_section.py').read_text().split('from google.colab import files')[0]
    exec(compile(export,'dashboard-export','exec'), context)
    print('Ready:',args.output/'dashboard_export.zip')


if __name__ == '__main__':
    main()
