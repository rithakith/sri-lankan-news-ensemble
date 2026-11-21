# 12. Frozen experiment dashboard export — code generated with ChatGPT assistance.
# Run after training, validation analysis and frozen-model test evaluation.
from pathlib import Path
import json, shutil, hashlib, importlib.metadata, platform
import pandas as pd
import numpy as np
import joblib

required = ['fitted_bases', 'stack_meta', 'ann_pipeline', 'stack_encoder', 'blend_weight',
            'stack_valid_ids', 'test_ids', 'test_predictions', 'predictions', 'stack_train_ids',
            'history', 'blend_search']
missing = [name for name in required if name not in globals()]
assert not missing, f'Complete training and evaluation first. Missing: {missing}'
export_dir = Path(OUTPUT_DIR) / 'dashboard_export'
export_dir.mkdir(exist_ok=True)
categories = stack_encoder.classes_.tolist()
assert set(stack_valid_ids).isdisjoint(test_ids)
assert set(stack_train_ids).isdisjoint(set(stack_valid_ids) | set(test_ids))
bundle = {'ann_pipeline': ann_pipeline, 'stack_base_pipelines': fitted_bases,
          'stack_meta': stack_meta, 'base_order': list(base_models),
          'label_encoder': stack_encoder, 'stack_weight': float(blend_weight),
          'input_policy': 'headline_plus_first_200_body_words'}
assert all(np.array_equal(m.classes_, np.arange(len(categories)))
           for m in [ann_pipeline, stack_meta, *fitted_bases.values()])
joblib.dump(bundle, export_dir / 'hybrid_bundle.joblib', compress=3)
for split, ids, preds in [('validation', stack_valid_ids, predictions), ('test', test_ids, test_predictions)]:
    out = pd.DataFrame({'doc_id': [rows[i]['doc_id'] for i in ids],
                        'headline': [rows[i]['description'] for i in ids],
                        'BART_reference': [rows[i]['predicted_topic'] for i in ids]})
    for name in ['LR', 'Random Forest', 'XGBoost', 'Stacking', 'ANN', 'Selected hybrid']:
        assert len(preds[name]) == len(ids)
        out[name] = stack_encoder.inverse_transform(np.asarray(preds[name], dtype=int))
    out.to_csv(export_dir / f'{split}_predictions.csv', index=False)
pd.DataFrame(history).to_csv(export_dir / 'ann_training_history.csv', index=False)
blend_search.to_csv(export_dir / 'blend_validation_search.csv', index=False)
support = pd.DataFrame({'Category': categories})
for split, ids in [('train', stack_train_ids), ('validation', stack_valid_ids), ('test', test_ids)]:
    counts = pd.Series([rows[i]['predicted_topic'] for i in ids]).value_counts()
    support[split] = [int(counts.get(cat, 0)) for cat in categories]
support.to_csv(export_dir / 'category_support.csv', index=False)
versions = {name: importlib.metadata.version(name)
            for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'joblib']}
run = {'mode': MODE, 'taxonomy_version': 'v8' if MODE == 'full_v8' else 'v7',
       'input_policy': bundle['input_policy'], 'categories': categories,
       'excluded_categories': list(excluded), 'input_articles': len(rows),
       'comparison_counts': {'train': len(stack_train_ids), 'validation': len(stack_valid_ids), 'test': len(test_ids)},
       'original_split_counts': {name: len(ids) for name, ids in assigned.items()},
       'stack_weight': float(blend_weight), 'ann_best_epoch': int(best_epoch), 'folds': int(n_folds),
       'seed': int(SEED), 'python_version': platform.python_version(), 'package_versions': versions,
       'reference': 'BART-generated labels; no independent human-labelled benchmark',
       'test_settings_frozen': True, 'input_sha256': hashlib.sha256(raw).hexdigest(),
       'source': 'https://github.com/nuuuwan/lk_news/tree/data'}
run['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in export_dir.iterdir()
                       if p.suffix in ['.csv', '.joblib']}
(export_dir / 'run.json').write_text(json.dumps(run, indent=2))
(export_dir / 'model-requirements.txt').write_text('\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n')
archive = shutil.make_archive(str(export_dir), 'zip', root_dir=export_dir)
print('Exported:', archive, '| Run:', MODE, '| Counts:', run['comparison_counts'])
from google.colab import files
files.download(archive)
