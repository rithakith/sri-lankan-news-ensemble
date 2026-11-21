"""Build lossless English news tables. Generated with assistance from ChatGPT.

Requires pyarrow. Run from the project root.
"""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

source = Path('data/lk_news-full.zip')
out = Path('data/english')
out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(source) as z:
    paths = {n.split('/')[-2]: n.rsplit('/', 1)[0] for n in z.namelist() if n.endswith('/doc.json')}
    manifest_path = next(n for n in z.namelist() if n.endswith('/data/lk_news/docs_all.tsv'))
    manifest = list(csv.DictReader(io.StringIO(z.read(manifest_path).decode('utf-8-sig')), delimiter='\t'))
    english = [r for r in manifest if r['lang'] == 'en']
    keys = list(manifest[0])
    records = []
    conflicts = []
    for m in english:
        p = paths[m['doc_id']]
        r = json.loads(z.read(p + '/doc.json'))
        for k in r:
            if k not in keys:
                keys.append(k)
        for k, v in m.items():
            if k not in r:
                r[k] = v
            elif str(r[k]) != v:
                conflicts.append((m['doc_id'], k))
        if r['lang'] != 'en':
            raise ValueError('Manifest and document language disagree')
        r['text'] = z.read(p + '/doc.txt').decode('utf-8')
        records.append(r)
    if conflicts:
        raise ValueError(f'Conflicting source metadata: {conflicts[:5]}')
    if 'text' in keys:
        raise ValueError('Source already has text attribute')
    keys.append('text')
    schema = pa.schema([pa.field(k, pa.float64() if k == 'time_ut' else pa.string()) for k in keys])
    if len({r['doc_id'] for r in records}) != len(records):
        raise ValueError('Duplicate document IDs')
    table = pa.Table.from_pylist(records, schema=schema)
    parquet_path = out / 'news_english.parquet'
    pq.write_table(table, parquet_path, compression='zstd')
    csv_path = out / 'news_english.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    # Verify every value and every full article after saving both formats.
    assert pq.read_table(parquet_path).equals(table)
    with csv_path.open(encoding='utf-8', newline='') as f:
        csv.field_size_limit(100_000_000)
        loaded = csv.DictReader(f)
        count = 0
        for original, saved in zip(records, loaded, strict=True):
            assert all(saved[k] == ('' if original.get(k) is None else str(original[k])) for k in keys)
            count += 1
        assert count == len(records)
    report = {
        'source': 'https://github.com/nuuuwan/lk_news/tree/data',
        'source_archive_sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
        'selection': "source lang == 'en'; no text cleaning or deduplication",
        'rows': len(records), 'columns': keys,
        'schema': {f.name: str(f.type) for f in schema},
        'sources': dict(Counter(r['newspaper_id'] for r in records)),
        'missing_text': sum(not r['text'].strip() for r in records),
        'duplicate_text_records': len(records) - len({r['text'] for r in records}),
        'date_min': min(r['date_str'] for r in records),
        'date_max': max(r['date_str'] for r in records),
        'verification': 'All Parquet and CSV values round-trip exactly; text joined by doc_id from doc.txt',
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()} for p in [parquet_path, csv_path]},
    }
    (out / 'dataset_info.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2))
