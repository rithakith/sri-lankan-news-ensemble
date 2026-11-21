import json
from pathlib import Path
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# LankaNews: zero-shot topic suggestions
Code generated with assistance from ChatGPT. Understand and adapt it before submission.

Model and example source: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli). This BART model was trained on MultiNLI, not on our Sri Lankan topic labels. We supply candidate descriptions and it ranks them using natural-language inference. No topic-training examples are needed for this step.

Independent NLI scores are computed for simple topic phrases. Each displayed category uses the maximum of its alternative phrase scores; the highest-scoring category is selected. These scores do not sum to one and are not calibrated accuracy. Categories with more alternatives have more opportunities for a high match, so this pooling rule needs validation.

These are **pseudo-labels**, not human-verified ground truth. Evaluate final ensembles on a separately, independently labelled test set. Candidate wording affects results. Low confidence is a review status, not the Other category.

Run the preprocessing notebook first. Its cleaned file is `/content/news_english_cleaned.csv`. Runtime files disappear when Colab resets; save inputs and outputs to Drive for persistence. For a full run, use Runtime → Change runtime type → T4 GPU before loading the model. Changing runtime may require reloading the CSV.
''')
code('''%pip -q install "transformers>=4.44,<5" accelerate
''')
code('''from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm

# Same runtime as preprocessing: this path already exists.
# Alternatively upload the cleaned CSV using Colab's Files panel and update this path.
INPUT_PATH = Path('/content/news_english_cleaned.csv')
df = pd.read_csv(INPUT_PATH, keep_default_na=False)
assert df['doc_id'].is_unique
assert df['text'].str.strip().ne('').all()
print(f'Loaded {len(df):,} cleaned articles.')
''')
md('''## 1. Load model and define topics
Eleven primary topics follow our provisional topic guide. Other means readable articles outside the ten core topics. The model cannot enforce all guide boundary rules, so review examples before committing to this taxonomy.

Use the headline and first 200 whitespace-separated body words. This limits runtime and focuses on the news lead, but can miss later context. The model also truncates inputs to its token limit.
''')
code('''MODEL_ID = 'facebook/bart-large-mnli'
TOPICS = {
    'politics_governance': 'politics and government',
    'business_economy': 'business and economics',
    'crime_justice': 'crime and court cases',
    'sport': 'sports',
    'health_environment': 'health, weather, environment or natural disasters',
    'transport': 'transport and vehicles',
    'telecommunications': 'telecommunications and internet connectivity',
    'information_technology': 'software, computing or artificial intelligence',
    'education': 'education',
    'entertainment': 'entertainment and show business',
    'other': 'lifestyle, culture, religion or general-interest news',
}
# Score simple alternative subjects, then take the strongest match per topic.
ALIASES = {
    'politics': 'politics_governance',
    'government and diplomacy': 'politics_governance',
    'business': 'business_economy',
    'economics': 'business_economy',
    'crime': 'crime_justice',
    'legal proceedings': 'crime_justice',
    'sports': 'sport',
    'cricket': 'sport',
    'football': 'sport',
    'medicine and healthcare': 'health_environment',
    'weather': 'health_environment',
    'environmental protection and pollution': 'health_environment',
    'natural disasters': 'health_environment',
    'transport': 'transport',
    'vehicles': 'transport',
    'telecommunications': 'telecommunications',
    'information technology': 'information_technology',
    'education': 'education',
    'entertainment': 'entertainment',
    'music': 'entertainment',
    'films and television': 'entertainment',
    'religion and spirituality': 'other',
    'lifestyle and human interest': 'other',
}
TEMPLATE = 'This text is about {}.'
classifier = pipeline('zero-shot-classification', model=MODEL_ID,
                      device=0 if torch.cuda.is_available() else -1)
REVISION = getattr(classifier.model.config, '_commit_hash', None) or 'unknown'
BATCH_SIZE = 8 if torch.cuda.is_available() else 1
print('Device:', classifier.device, '| model revision:', REVISION)

# Heuristics for review, NOT validated accuracy thresholds.
MIN_SCORE = 0.70
MIN_MARGIN = 0.20

def predict_topics(frame):
    inputs = [str(title) + '\\n' + ' '.join(str(text).split()[:200])
              for title, text in zip(frame['description'], frame['text'])]
    results = classifier(inputs, candidate_labels=list(ALIASES),
                         hypothesis_template=TEMPLATE, multi_label=True,
                         batch_size=BATCH_SIZE)
    rows = []
    for doc_id, result in zip(frame['doc_id'], results):
        alias_scores = dict(zip(result['labels'], result['scores']))
        scores = {topic: max(alias_scores[a] for a, group in ALIASES.items() if group == topic)
                  for topic in TOPICS}
        ranked = sorted(scores, key=scores.get, reverse=True)
        score, second_score = scores[ranked[0]], scores[ranked[1]]
        row = {
            'doc_id': doc_id,
            'predicted_topic': ranked[0],
            'matched_phrase': max((a for a, group in ALIASES.items() if group == ranked[0]), key=alias_scores.get),
            'teacher_score': score,
            'teacher_margin': score - second_score,
            'needs_review': score < MIN_SCORE or score - second_score < MIN_MARGIN,
            'label_source': 'zero_shot_pseudo_label',
            'teacher_model': MODEL_ID,
            'teacher_revision': REVISION,
            'taxonomy_version': 'v7',
            'input_policy': 'headline_plus_first_200_body_words',
            'scoring_policy': 'independent_alias_scores_max_per_topic',
        }
        row.update({'score_' + topic: scores[topic] for topic in TOPICS})
        rows.append(row)
    return pd.DataFrame(rows)
''')
md('''## 2. Pilot on 100 articles
A reproducible random sample is a smoke test, not an accuracy evaluation. Inspect the headline and body, especially cases where politics overlaps economics or crime. Change taxonomy only using development examples, not your final independent test set.
''')
code('''pilot = df.sample(n=min(100, len(df)), random_state=42)
pilot_predictions = predict_topics(pilot)
pilot_output = pilot.merge(pilot_predictions, on='doc_id', validate='one_to_one')
pilot_output.to_csv('/content/news_zero_shot_pilot_v7_100.csv', index=False)
display(pilot_output[['description', 'predicted_topic', 'teacher_score',
                      'teacher_margin', 'needs_review']])
print(pilot_output['predicted_topic'].value_counts())
''')
cells = cells[:7]
md('Only the 100-row pilot runs in this version. Education is a separate category (taxonomy v7); previous full-dataset jobs were stopped. Output retains all original columns and eleven category scores.')
code("assert len(pilot_output) == 100\nprint('Saved exactly 100 articles.')\nfrom google.colab import files\nfiles.download('/content/news_zero_shot_pilot_v7_100.csv')\n")
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='LankaNews_ZeroShot.ipynb'),kernelspec=dict(name='python3',display_name='Python 3'),language_info=dict(name='python')),cells=cells)
p=Path('notebooks/LankaNews_ZeroShot.ipynb'); p.write_text(json.dumps(nb,indent=2))
# Validate syntax of normal Python cells (the install cell is IPython syntax).
for c in cells:
    if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'):
        compile(''.join(c['source']),str(p),'exec')
print(p, 'created; Python cell syntax checked')
