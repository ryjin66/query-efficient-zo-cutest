"""Verify distributed files without importing scientific dependencies."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--environment-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'checksums.json').read_text())
    count = 0
    for relative, expected in manifest.items():
        if args.environment_only and not relative.startswith('environment/'):
            continue
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe release path: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Checksum mismatch: '+relative)
        count += 1
    print(f'Verified {count} files.')


if __name__ == '__main__':
    main()
