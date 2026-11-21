"""Install the notebook export ZIP. Does not execute serialized models. ChatGPT-assisted."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'hybrid_bundle.joblib', 'run.json', 'validation_predictions.csv', 'test_predictions.csv',
           'ann_training_history.csv', 'blend_validation_search.csv', 'category_support.csv', 'model-requirements.txt'}


def install(archive, destination, update_requirements=True):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if set(names) - ALLOWED or len(set(names)) != len(names):
            raise ValueError('ZIP contains unexpected paths or duplicate entries.')
        required = {'run.json', 'hybrid_bundle.joblib', 'validation_predictions.csv', 'test_predictions.csv', 'model-requirements.txt'}
        if not required <= set(names):
            raise ValueError('Incomplete experiment export.')
        if sum(i.file_size for i in z.infolist()) > 750_000_000:
            raise ValueError('Export exceeds 750 MB uncompressed.')
        run = json.loads(z.read('run.json'))
        hashes = run.get('files_sha256', {})
        for name in required - {'run.json', 'model-requirements.txt'}:
            if name not in hashes:
                raise ValueError(f'Missing checksum for {name}')
        for name, digest in hashes.items():
            if name not in names or hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError(f'Checksum mismatch: {name}')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if any(p.name in ALLOWED for p in destination.iterdir()):
            raise ValueError('Destination contains an experiment. Use a new directory to preserve it.')
        for name in names:
            (destination / name).write_bytes(z.read(name))
        if update_requirements:
            (ROOT / 'dashboard' / 'requirements.txt').write_text(
                'streamlit==1.67.0\nplotly==6.9.0\n-r artifacts/model-requirements.txt\n')
        print(f"Installed {run['mode']} export into {destination}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'dashboard' / 'artifacts')
    args = parser.parse_args()
    install(args.archive, args.destination)
