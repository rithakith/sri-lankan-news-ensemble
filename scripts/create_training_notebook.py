"""Rebuild the training notebook from the sectioned Python source."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'notebooks/LankaNews_Ensemble_Training_10000.py'
cells = []
for block in source.read_text().split('# %%'):
    if not block.strip():
        continue
    markdown = block.startswith(' [markdown]')
    lines = block.splitlines()[1:]
    text = '\n'.join(line[2:] if line.startswith('# ') else line[1:] if line == '#' else line for line in lines) if markdown else '\n'.join(lines)
    cell = {'cell_type': 'markdown' if markdown else 'code', 'metadata': {}, 'source': text.strip().splitlines(True)}
    if not markdown:
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
# Retain deployment export when regenerating either training notebook.
export_source = (ROOT / 'scripts/export_dashboard_section.py').read_text()
if not any('Frozen experiment dashboard export' in ''.join(cell['source']) for cell in cells):
    cells.extend([
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 12. Dashboard export\nPackage the frozen experiment and matching evaluation files. ChatGPT-assisted code.']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': export_source.splitlines(True)},
    ])
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'colab': {'name': 'LankaNews_Ensemble_Training_10000.ipynb'}}, 'nbformat': 4, 'nbformat_minor': 5}
for filename in ('LankaNews_Ensemble_Training_10000.ipynb', 'LankaNews_Ensemble_Training.ipynb'):
    (ROOT / 'notebooks' / filename).write_text(json.dumps(nb, indent=1))
