#!/usr/bin/env python3
"""Fetch a minimal official source snapshot, verifying every Git blob hash.

Avoids large, unused RADIO images and COCO annotations. No modifications to upstream.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '6da781bf02046fe6f70530e1d5483abd9aef6c9c'
REPO = 'Intellindust-AI-Lab/FSOD-VFM'


def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tree_path = Path('artifacts/baseline_tree.json')
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
    else:
        tree = json.load(opener.open(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1', timeout=30))
    if tree.get('sha') != COMMIT or tree.get('truncated'):
        raise ValueError('Unexpected or incomplete source tree')
    selected = [x for x in tree['tree'] if x['type'] == 'blob' and
                (x['path'].startswith(('chatrex/', 'model/', 'run_scripts/', 'data/CDFSOD/'))
                 or x['path'] in ('main.py', 'support_util.py', 'query_util.py', 'metric.py',
                                  'README.md', 'LICENSE', 'fsod.yml'))]
    root = Path('external/FSOD-VFM')
    def fetch(item):
        path = root / item['path']; path.parent.mkdir(parents=True, exist_ok=True)
        def valid(body):
            return hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest() == item['sha']
        if path.exists() and valid(path.read_bytes()):return item
        for attempt in range(3):
            try:
                body = opener.open(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item["path"]}', timeout=30).read()
                if not valid(body):raise ValueError('Git blob hash mismatch')
                path.write_bytes(body);return item
            except Exception:
                if attempt == 2:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(fetch, selected))
    manifest = {'repository': f'https://github.com/{REPO}', 'commit': COMMIT,
                'format': 'git-blob-verified source subset', 'files': files}
    (root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Verified {len(files)} upstream files at {COMMIT}')


if __name__ == '__main__':main()
