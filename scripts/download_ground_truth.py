import argparse
from pathlib import Path
import zipfile
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', required=True, help='Pinned Hugging Face dataset commit')
    parser.add_argument('--output', default='data/raw')
    args = parser.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(repo_id='BGLab/MaizeField3D', repo_type='dataset',
        filename='FielGrwon_ZeaMays_SegmentedPCD_100k.zip', revision=args.revision)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe archive path')
        z.extractall(root)

if __name__ == '__main__':
    main()
