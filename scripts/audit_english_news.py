"""Read-only source audit; generated with assistance from ChatGPT.

Requires pyarrow. Produces review flags, not verified language/topic labels.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlparse
import pyarrow as pa
import pyarrow.parquet as pq

root = Path('data/english')
out = root / 'audit'
out.mkdir(exist_ok=True)
rows = pq.read_table(root / 'news_english.parquet').to_pylist()
sections = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
months = collections.defaultdict(collections.Counter)
text_groups = collections.defaultdict(list)
title_groups = collections.defaultdict(list)
index = []
flags = collections.Counter()
business_sections = {'business', 'business-news', 'business-main'}
for r in sorted(rows, key=lambda r: (r['date_str'], r['doc_id'])):
    text = r['text']
    norm = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
    digest = hashlib.sha256(norm.encode()).hexdigest()
    title_norm = re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', r['description']).casefold()).strip()
    title_hash = hashlib.sha256(title_norm.encode()).hexdigest()
    words = len(text.split())
    letters = [c for c in text if c.isalpha()]
    latin_ratio = sum('LATIN' in unicodedata.name(c, '') for c in letters) / max(len(letters), 1)
    path = urlparse(r['url_metadata']).path.strip('/').split('/')
    section = unquote(path[0]) if path else ''
    # Most publishers put the unique story slug first; those aren't categories.
    source = r['newspaper_id']
    meaningful_section = section if source in {'dailymirrorlk', 'dailyftlk'} else None
    sections[source][section if meaningful_section else '(no section in URL)'] += 1
    if len(examples[source, section if meaningful_section else '(no section in URL)']) < 3:
        examples[source, section if meaningful_section else '(no section in URL)'].append({'doc_id': r['doc_id'], 'headline': r['description'], 'url': r['url_metadata']})
    months[source][r['date_str'][:7]] += 1
    duplicate_of = text_groups[digest][0] if text_groups[digest] else None
    text_groups[digest].append(r['doc_id'])
    if title_norm:
        title_groups[title_hash].append(r['doc_id'])
    issues = []
    if words < 50: issues.append('short_text_under_50_words')
    if latin_ratio < .8: issues.append('non_latin_letters_over_20pct')
    if '\ufffd' in text: issues.append('unicode_replacement_character')
    if duplicate_of: issues.append('normalized_exact_duplicate')
    flags.update(issues)
    item = {k: v for k, v in r.items() if k != 'text'}
    item.update(text_characters=len(text), text_words=words, latin_letter_ratio=latin_ratio,
                url_section=meaningful_section, text_sha256=digest, title_group=title_hash,
                duplicate_of=duplicate_of, review_flags=issues,
                pilot_candidate=not bool(issues),
                weak_business_section=bool(meaningful_section in business_sections))
    index.append(item)

pq.write_table(pa.Table.from_pylist(index), out / 'article_index.parquet', compression='zstd')
for filename, groups in [('exact_duplicate_groups.json', text_groups), ('same_headline_candidates.json', title_groups)]:
    (out / filename).write_text(json.dumps({k:v for k,v in groups.items() if len(v)>1}, indent=2))

report = {'articles': len(rows), 'unique_doc_ids': len({r['doc_id'] for r in rows}),
          'sources': dict(collections.Counter(r['newspaper_id'] for r in rows)),
          'date_min': min(r['date_str'] for r in rows), 'date_max': max(r['date_str'] for r in rows),
          'review_flags': dict(flags), 'pilot_candidates': sum(r['pilot_candidate'] for r in index),
          'normalized_unique_texts': len(text_groups),
          'exact_duplicate_groups': sum(len(v)>1 for v in text_groups.values()),
          'same_headline_groups': sum(len(v)>1 for v in title_groups.values()),
          'same_headline_extra_records': sum(len(v)-1 for v in title_groups.values() if len(v)>1),
          'weak_business_section_records': sum(r['weak_business_section'] for r in index),
          'sections': {s:dict(c) for s,c in sections.items()},
          'monthly_coverage': {s:dict(sorted(c.items())) for s,c in months.items()},
          'examples': {s:{sec:ex for (src,sec),ex in examples.items() if src==s} for s in sections},
          'flagged_examples': [r for r in index if r['review_flags']][:40],
          'limitations': ['Language flag is a Unicode-script heuristic, not language verification.',
                         'Same-headline groups are review candidates, not confirmed duplicate stories.',
                         'Semantic near-duplicates and syndicated rewrites have not been exhaustively detected.',
                         'Publisher sections provide weak evidence, not verified topic labels.',
                         'No records have been deleted and no final topic labels assigned.']}
(out / 'audit_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'examples','sections','monthly_coverage','flagged_examples'}}, indent=2))
print('NON_LATIN_EXAMPLES')
for r in index:
    if r['latin_letter_ratio'] < .8:
        print(r['doc_id'], r['description'][:140], round(r['latin_letter_ratio'],2))
