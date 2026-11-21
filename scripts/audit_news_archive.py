"""Audit the complete upstream news snapshot without extracting all files.

Generated with assistance from ChatGPT.
"""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse
import zipfile

archive = Path('data/lk_news-full.zip')
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    manifest = next(n for n in names if n.endswith('/data/lk_news/docs_all.tsv'))
    rows = list(csv.DictReader(io.StringIO(z.read(manifest).decode('utf-8-sig')), delimiter='\t'))
    text_files = {n.rsplit('/', 1)[0]: z.getinfo(n).file_size for n in names if n.endswith('/doc.txt')}
    doc_paths = {n.split('/')[-2]: n.rsplit('/', 1)[0] for n in names if n.endswith('/doc.json')}
    counts = collections.Counter(r.get('lang') for r in rows)
    sources = {}
    for r in rows:
        if r.get('lang') != 'en':
            continue
        source = r.get('newspaper_id', 'unknown')
        item = sources.setdefault(source, {'articles': 0, 'with_nonempty_text': 0, 'domains': set(), 'first_date': r['date_str'], 'last_date': r['date_str'], 'example_url': r.get('url_metadata')})
        item['articles'] += 1
        item['first_date'] = min(item['first_date'], r['date_str'])
        item['last_date'] = max(item['last_date'], r['date_str'])
        item['domains'].add(urlparse(r.get('url_metadata', '')).netloc)
        path = doc_paths.get(r.get('doc_id'))
        item['with_nonempty_text'] += bool(path and text_files.get(path, 0) > 0)
    for item in sources.values():
        item['domains'] = sorted(item['domains'])
    print('Checking archive CRCs...', flush=True)
    bad = z.testzip()
    if bad:
        raise RuntimeError(f'Corrupt archive entry: {bad}')
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    result = {'source_url': 'https://github.com/nuuuwan/lk_news/tree/data', 'sha256': digest, 'compressed_bytes': archive.stat().st_size, 'uncompressed_bytes': sum(i.file_size for i in z.infolist()), 'manifest_records': len(rows), 'document_json_files': len(doc_paths), 'text_files': len(text_files), 'languages': dict(counts), 'english_sources': dict(sorted(sources.items(), key=lambda kv: -kv[1]['articles'])), 'crc_check': 'passed'}
    Path('data/news-access-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    Path('data/news-documents.tsv').write_bytes(z.read(manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
