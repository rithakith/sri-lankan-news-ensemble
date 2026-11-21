"""Create a portable Colab preprocessing notebook. Assisted by ChatGPT."""
import json
from pathlib import Path

load = '''from pathlib import Path
import pandas as pd
from google.colab import files

# Upload news_english.csv (the original English dataset).
uploaded = files.upload()
csv_paths = [name for name in uploaded if name.lower().endswith('.csv')]
if len(csv_paths) != 1:
    raise ValueError("Upload exactly one source CSV.")
INPUT_PATH = Path(csv_paths[0])
del uploaded  # Release the extra in-memory upload copy.

df = pd.read_csv(INPUT_PATH, keep_default_na=False)
print(f"Loaded {len(df):,} articles and {len(df.columns)} columns.")
display(df.drop(columns=['text']).head())
'''
clean = '''# Generated with assistance from ChatGPT; reviewed/adapted for this assignment.
import hashlib
import re
import unicodedata

MIN_WORDS = 50
MIN_LATIN_RATIO = 0.80
original_columns = df.columns.tolist()
required = {'doc_id', 'date_str', 'lang', 'text'}
assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
assert df['doc_id'].is_unique, "Review duplicate document IDs before continuing."

def normalized_text_hash(text):
    normalized = unicodedata.normalize('NFKC', str(text))
    normalized = re.sub(r'\\s+', ' ', normalized).strip().casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def latin_letter_ratio(text):
    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    latin_letters = sum('LATIN' in unicodedata.name(char, '') for char in letters)
    return latin_letters / len(letters)

# Sort deterministically: retain the earliest record in each exact duplicate group.
working = df.sort_values(['date_str', 'doc_id'], kind='stable').copy()
word_counts = working['text'].str.split().str.len()
latin_ratios = working['text'].map(latin_letter_ratio)
text_hashes = working['text'].map(normalized_text_hash)

# These are reproducible quality-screening rules, not perfect language detection.
flags = pd.DataFrame({
    'not_tagged_english': working['lang'].ne('en'),
    'short_text': word_counts.lt(MIN_WORDS),
    'language_review': latin_ratios.lt(MIN_LATIN_RATIO),
    'replacement_character': working['text'].str.contains('\\ufffd', regex=False),
    'normalized_exact_duplicate': text_hashes.duplicated(keep='first'),
}, index=working.index)

summary = flags.sum().rename('flagged_articles').to_frame()
print("Flag counts can overlap:")
display(summary)

remove_mask = flags.any(axis=1)
cleaned_df = working.loc[~remove_mask, original_columns].reset_index(drop=True)
print(f"Input:   {len(df):,}")
print(f"Removed: {int(remove_mask.sum()):,} unique articles")
print(f"Kept:    {len(cleaned_df):,}")

# Preserve source text and every original attribute; only filter rows.
assert cleaned_df.columns.tolist() == original_columns
assert cleaned_df['doc_id'].is_unique
assert cleaned_df['lang'].eq('en').all()
assert not cleaned_df['text'].map(normalized_text_hash).duplicated().any()
assert len(cleaned_df) + int(remove_mask.sum()) == len(df)

OUTPUT_PATH = Path('news_english_cleaned.csv')
cleaned_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
saved = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
pd.testing.assert_frame_equal(saved, cleaned_df)
print(f"Saved and verified: {OUTPUT_PATH.resolve()}")
'''
download = '''from google.colab import files
files.download(str(OUTPUT_PATH))
'''
def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(keepends=True)}
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'LankaNews_Preprocessing.ipynb'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
md('# LankaNews: reproducible preprocessing\n\nUpload the original `news_english.csv`, filter flagged articles, and save a separate cleaned CSV. All original columns and retained article texts are preserved. Code generated with assistance from ChatGPT.\n\nRules: remove articles under 50 words, those with less than 80% Latin alphabetic characters, text containing the Unicode replacement character, and normalized exact-text repeats (keep earliest). These are screening choices, not proof of invalidity. Same-headline and semantic near-duplicates are not automatically removed. No stemming, stop-word removal or topic labelling is performed.'),
md('## 1. Load the source CSV'),code(load),md('## 2. Filter flagged records and save a cleaned CSV'),code(clean),md('## 3. Download the cleaned file\n\nColab runtime storage is temporary. Download the result or copy it to your Drive.'),code(download)]}
Path('notebooks').mkdir(exist_ok=True)
Path('notebooks/LankaNews_Preprocessing.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
Path('/tmp/lankanews-clean-cell.py').write_text(clean)
