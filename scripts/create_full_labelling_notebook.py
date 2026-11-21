import json
from pathlib import Path
old=json.loads(Path('notebooks/LankaNews_ZeroShot.ipynb').read_text())
config=''.join(old['cells'][4]['source'])
config=config.replace("'health_environment': 'health, weather, environment or natural disasters',", "'health_healthcare': 'Health & Healthcare',\n    'weather_environment_disasters': 'Weather, Environment & Disasters',\n    'tourism_travel': 'Tourism & Travel',")
config=config.replace("'medicine and healthcare': 'health_environment'","'medicine and healthcare': 'health_healthcare'").replace("'health_environment'","'weather_environment_disasters'")
config=config.replace("    'religion and spirituality': 'other',","    'tourism and travel': 'tourism_travel',\n    'hotels and holiday destinations': 'tourism_travel',\n    'religion and spirituality': 'other',")
config=config.replace("classifier = pipeline('zero-shot-classification', model=MODEL_ID,\n                      device=0 if torch.cuda.is_available() else -1)","""MODEL_REVISION = 'd7645e127eaf1aefc7862fd59a17a5aa8558b8ce'
assert torch.cuda.is_available(), 'Select a Colab GPU runtime before the full run.'
classifier = pipeline('zero-shot-classification', model=MODEL_ID,
                      revision=MODEL_REVISION, device=0, torch_dtype=torch.float16)""")
config=config.replace('BATCH_SIZE = 8 if torch.cuda.is_available() else 1','BATCH_SIZE = 16')
config=config.replace("    results = classifier(inputs, candidate_labels=list(ALIASES),\n                         hypothesis_template=TEMPLATE, multi_label=True,\n                         batch_size=BATCH_SIZE)","""    global BATCH_SIZE
    while True:
        try:
            results = classifier(inputs, candidate_labels=list(ALIASES),
                                 hypothesis_template=TEMPLATE, multi_label=True,
                                 batch_size=BATCH_SIZE)
            break
        except torch.cuda.OutOfMemoryError:
            if BATCH_SIZE == 1:
                raise
            BATCH_SIZE = max(1, BATCH_SIZE // 2)
            torch.cuda.empty_cache()
            print('Reduced GPU batch size to', BATCH_SIZE, flush=True)""")
config=config.replace("'taxonomy_version': 'v7'","'taxonomy_version': 'v8'")
config += "\nassert len(TOPICS) == 13\nassert set(ALIASES.values()) == set(TOPICS)\n"
setup="""from pathlib import Path
import pandas as pd
import torch
import numpy as np
import json, hashlib, os, time
from transformers import pipeline
from tqdm.auto import tqdm
INPUT_PATH = Path('/content/news_english_cleaned.csv')
df = pd.read_csv(INPUT_PATH, keep_default_na=False)
assert len(df) == 50528 and df['doc_id'].is_unique
assert df['text'].str.strip().ne('').all()
print(f'Loaded {len(df):,} cleaned articles; preserving {len(df.columns)} original columns.')
"""
run="""# Saves each completed 100-row chunk atomically; safe to rerun with the same input/config.
# Existing mounted Drive is preferred. Otherwise /content is temporary runtime storage.
if Path('/content/drive/MyDrive').exists():
    RUN_DIR = Path('/content/drive/MyDrive/LankaNews/zero_shot_v8')
else:
    RUN_DIR = Path('/content/lankanews_v8')
RUN_DIR.mkdir(parents=True, exist_ok=True)
PARTS = RUN_DIR / 'checkpoints'
PARTS.mkdir(exist_ok=True)
OUTPUT = RUN_DIR / 'news_english_pseudo_labelled_v8.csv'

input_hash = hashlib.sha256()
with INPUT_PATH.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        input_hash.update(block)
settings = {'model': MODEL_ID, 'revision': REVISION, 'topics': TOPICS, 'aliases': ALIASES,
            'template': TEMPLATE, 'taxonomy': 'v8', 'input_sha256': input_hash.hexdigest(),
            'input_policy': 'headline_plus_first_200_body_words', 'precision': 'float16',
            'min_score': MIN_SCORE, 'min_margin': MIN_MARGIN,
            'scoring_policy': 'independent_alias_scores_max_per_topic', 'chunk_size': 100}
config_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
manifest_path = RUN_DIR / 'run_settings.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == settings, 'Input/config changed: use a new run directory.'
else:
    manifest_path.write_text(json.dumps(settings, indent=2))

# Small execution check before launching all articles; not an accuracy evaluation.
smoke = predict_topics(df.iloc[:3])
score_columns = ['score_' + topic for topic in TOPICS]
assert np.isfinite(smoke[score_columns].to_numpy()).all()
assert smoke['predicted_topic'].isin(TOPICS).all()
print('GPU execution check passed. Categories:', len(TOPICS), '| device:', classifier.device, flush=True)
print('Checkpoint/output folder:', RUN_DIR, flush=True)
print('Storage:', 'Google Drive' if '/drive/MyDrive/' in str(RUN_DIR) else 'temporary Colab runtime; download results before resetting', flush=True)

started = time.monotonic()
for start in tqdm(range(0, len(df), 100), desc='Full dataset (100 articles per chunk)'):
    frame = df.iloc[start:start + 100]
    part = PARTS / f'part_{start:06d}.csv'
    if part.exists():
        saved = pd.read_csv(part, keep_default_na=False)
        assert saved['doc_id'].tolist() == frame['doc_id'].tolist()
        assert saved['run_config_id'].eq(config_id).all()
        assert np.isfinite(saved[score_columns].to_numpy()).all()
        continue
    result = predict_topics(frame)
    assert len(result) == len(frame)
    assert result['doc_id'].tolist() == frame['doc_id'].tolist()
    assert np.isfinite(result[score_columns].to_numpy()).all()
    result['run_config_id'] = config_id
    temp = part.with_suffix('.csv.tmp')
    result.to_csv(temp, index=False)
    os.replace(temp, part)
    completed = min(start + 100, len(df))
    if start == 0 or completed % 1000 == 0 or completed == len(df):
        print(f'Checkpointed {completed:,}/{len(df):,} articles; elapsed {(time.monotonic()-started)/60:.1f} min', flush=True)

predictions = pd.concat([pd.read_csv(p, keep_default_na=False) for p in sorted(PARTS.glob('part_*.csv'))], ignore_index=True)
assert len(predictions) == len(df) and predictions['doc_id'].is_unique
assert set(predictions['doc_id']) == set(df['doc_id'])
assert predictions['predicted_topic'].isin(TOPICS).all()
labelled_df = df.merge(predictions, on='doc_id', how='left', validate='one_to_one')
assert labelled_df[df.columns].equals(df)
labelled_df.to_csv(OUTPUT, index=False)
labelled_df.to_csv(OUTPUT.with_suffix('.csv.gz'), index=False, compression='gzip')
print('COMPLETE:', len(labelled_df), 'articles saved to', OUTPUT, flush=True)
display(labelled_df['predicted_topic'].value_counts().rename('articles').to_frame())
print('Review flags:', labelled_df['needs_review'].sum())
"""
intro="""# LankaNews: full-dataset pseudo-labelling — taxonomy v8
Generated with assistance from ChatGPT. Model: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), pinned revision. This is inference, not model training. Labels are model-generated references, not verified facts or topic ground truth.

13 categories: Politics & Governance; Business & Economy; Crime & Justice; Sports; Health & Healthcare; Weather, Environment & Disasters; Transport & Mobility; Telecommunications; Information Technology; Education; Entertainment & Arts; Tourism & Travel; Other / General Interest.

All 50,528 cleaned articles are processed. Input is headline plus first 200 body words; the full original article is preserved in the exported CSV. Short phrases are scored independently, then max-pooled per category. Scores are not calibrated and do not sum to one. More alternatives can favour a category. Review flags are provisional. Category guide boundaries cannot be guaranteed by BART.

Uses the Colab GPU in float16 with automatic batch-size reduction on GPU memory exhaustion. Small numeric differences from float32 pilot scores are possible. Checkpoints are isolated by taxonomy, model, input hash and settings. Each completed chunk contains 100 predictions. If Google Drive is already mounted it is used; otherwise files are in temporary /content storage. A reset can erase runtime files. Resume requires the same input, settings and saved checkpoint folder.
"""
def cell(kind,s):
 c={'cell_type':kind,'metadata':{},'source':s.splitlines(True)}
 if kind=='code':c.update(execution_count=None,outputs=[])
 return c
cells=[cell('markdown',intro),cell('code','%pip -q install "transformers>=4.44,<5" accelerate\n'),cell('code',setup),cell('code',config),cell('code',run),cell('code',"from google.colab import files\nfiles.download(str(OUTPUT.with_suffix('.csv.gz')))\n")]
for c in cells:
 if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'):compile(''.join(c['source']),'full_labelling','exec')
nb={'nbformat':4,'nbformat_minor':5,'metadata':old['metadata'],'cells':cells}
nb['metadata']['colab']['name']='LankaNews_FullLabelling_v8.ipynb'
Path('notebooks/LankaNews_FullLabelling_v8.ipynb').write_text(json.dumps(nb,indent=2))
print('Full labelling notebook created and syntax verified.')
