"""Check the release files and recount saved results, without external packages."""
import hashlib
import json
from pathlib import Path

from summarize_saved_results import summarize


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            raise SystemExit(f"Missing file: {entry['path']}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise SystemExit(f"File differs from this release: {entry['path']}")
    expected = json.loads((root / 'RESULTS_SUMMARY.json').read_text())
    if summarize(root / 'artifact') != expected:
        raise SystemExit('Saved-result counts differ from RESULTS_SUMMARY.json')
    print(f"PASS: {len(manifest['files'])} release files match their hashes.")
    print('PASS: saved-result counts match RESULTS_SUMMARY.json.')


if __name__ == '__main__':
    main()
