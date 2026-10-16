#!/usr/bin/env python3
"""Download a minimal pinned SAM2 package with Git blob verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'facebookresearch/sam2'
COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'


def main():
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tree = json.load(op.open(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1', timeout=30))
    if tree.get('truncated'):raise ValueError('Truncated source tree')
    files = [i for i in tree['tree'] if i['type'] == 'blob' and
             (i['path'].startswith('sam2/') or i['path'] in
              ['setup.py', 'pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE', 'LICENSE_cctorch'])]
    root = Path('external/sam2');root.mkdir(parents=True, exist_ok=True)
    def fetch(i):
        path = root/i['path'];path.parent.mkdir(parents=True, exist_ok=True)
        def valid(body):
            return hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest() == i['sha']
        if path.exists() and valid(path.read_bytes()):return
        for attempt in range(3):
            try:
                body = op.open(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{i["path"]}', timeout=30).read()
                if not valid(body):raise ValueError('Git blob mismatch')
                path.write_bytes(body);return
            except Exception:
                if attempt == 2:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, files))
    (root/'source_manifest.json').write_text(json.dumps({'repository': REPO, 'commit': COMMIT,
                                                       'files': files}, indent=2))
    print(f'Verified {len(files)} SAM2 source files at {COMMIT}')


if __name__ == '__main__':main()
