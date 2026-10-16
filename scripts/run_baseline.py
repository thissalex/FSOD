#!/usr/bin/env python3
"""Fail-closed preflight, then run official main.py with optional support adapter."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsod_experiments.records import digest, file_digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--support', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--query-images', required=True)
    p.add_argument('--shot', required=True, type=int)
    p.add_argument('--domain', required=True)
    p.add_argument('--role', choices=['development', 'official_comparison', 'confirmatory'], required=True)
    p.add_argument('--method', choices=['B0', 'B1', 'B2', 'B5', 'Ours'], default='B0')
    p.add_argument('--lam', type=float, default=10)
    p.add_argument('--mu', type=float, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-id')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    run_id = args.run_id or datetime.now().strftime('%Y%m%dT%H%M%S') + f'-{args.domain}-{args.shot}s-{args.method}'
    out = ROOT / 'results' / run_id
    out.mkdir(parents=True, exist_ok=False)
    upstream = ROOT / 'external/FSOD-VFM'
    paths = {'support': Path(args.support).resolve(), 'query': Path(args.query).resolve(),
             'data_root': Path(args.data_root).resolve(), 'query_images': Path(args.query_images).resolve()}
    required = [upstream / 'source_manifest.json', upstream / 'main.py',
                upstream / 'dinov2/hubconf.py', ROOT / 'checkpoints/upn_large.pth',
                ROOT / 'checkpoints/dinov2_vitl14_pretrain.pth',
                ROOT / 'checkpoints/sam2.1_hiera_large.pt', *paths.values()]
    errors = [f'Missing: {path}' for path in required if not path.exists()]
    deps = ['torch', 'torchvision', 'sam2', 'MultiScaleDeformableAttention',
            'pycocotools.coco', 'mmengine', 'timm', 'ensemble_boxes', 'cv2']
    versions = {}
    for module in deps:
        check = subprocess.run([sys.executable, '-c',
                                f'import torch; import {module}; print(getattr({module.split(".")[0]}, "__version__", "import_ok"))'],
                               capture_output=True, text=True, timeout=60)
        versions[module] = check.stdout.strip()
        if check.returncode:
            errors.append(f'Import {module}: {check.stderr.strip().splitlines()[-1]}')
    record = {'run_id': run_id, 'status': 'preflight', 'baseline_commit':
              '6da781bf02046fe6f70530e1d5483abd9aef6c9c', 'args': vars(args),
              'python': sys.version, 'dependencies': versions, 'errors': errors,
              'metrics': None, 'query_cache': 'not_implemented'}
    support = None
    if paths['support'].is_file():
        support = json.loads(paths['support'].read_text())
        if not support or any(len(v) != args.shot for v in support.values()):
            official = upstream / f'data/CDFSOD/{args.domain}/annotations/{args.shot}_shot_converted.json'
            official_exception = (args.role == 'official_comparison' and official.is_file()
                                  and digest(support) == digest(json.loads(official.read_text())))
            if official_exception:
                record['shot_count_disclosure'] = {k: len(v) for k, v in support.items()}
            else:
                errors.append('Support set does not contain exactly K instances for every class')
        for samples in support.values():
            for sample in samples:
                if not (paths['data_root'] / sample['image']).is_file():
                    errors.append(f'Missing support image: {sample["image"]}')
        record['support_manifest_hash'] = digest(support)
    if paths['query'].is_file():
        query = json.loads(paths['query'].read_text())
        record['query_manifest_hash'] = digest(query)
        if support and set(support) != {c['name'] for c in query['categories']}:
            errors.append('Support/query category-name mismatch')
        for image in query['images']:
            if not (paths['query_images'] / image['file_name']).is_file():
                errors.append(f'Missing query image: {image["file_name"]}')
    record['status'] = 'blocked' if errors else 'ready'
    write_json(out / 'run.json', record)
    if errors:
        print('\n'.join(errors)); print(f'Preflight saved: {out}/run.json')
        return 2
    # Check all snapshot files; fail if anyone changed baseline implementation.
    import hashlib
    manifest = json.loads((upstream / 'source_manifest.json').read_text())
    for item in manifest['files']:
        body = (upstream / item['path']).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if actual != item['sha']:
            raise ValueError(f'Upstream source modified: {item["path"]}')
    record['model_weights_hash'] = {f.name: file_digest(f) for f in required[3:6]}
    record['feature_cache_key'] = digest({'baseline_commit':record['baseline_commit'],
                                         'weights':record['model_weights_hash'],
                                         'preprocessing':'official_DINOv2_630_fp32',
                                         'proposal_threshold':0.01, 'max_proposals':100,
                                         'sam2':'large_original_masks'})
    record['postprocess_config_hash'] = digest({'baseline_commit':record['baseline_commit'],
                                                'diffusion_steps':30, 'alpha':0.3, 'decay':0.5})
    record['implementation_hash'] = digest({str(f.relative_to(ROOT)): file_digest(f)
        for folder in ['fsod_experiments', 'scripts'] for f in (ROOT/folder).glob('*.py')})
    if args.role != 'development' and args.method != 'B0':
        raise ValueError('Formal adaptation runs require a future locked-configuration acceptance step')
    if args.preflight_only:
        write_json(out / 'run.json', record); print('Preflight passed'); return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; official code requires CUDA')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    record['status'] = 'running'; write_json(out / 'run.json', record)
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    import support_util
    import metric
    original_eval = metric.run_coco_eval
    def capture_eval(*eval_args, **eval_kwargs):
        evaluated = original_eval(*eval_args, **eval_kwargs)
        write_json(out / 'evaluation.json', evaluated)
        stats = evaluated['bbox']['stats']
        record['metrics'] = {'ap': 100*stats[0], 'ap50': 100*stats[1], 'ap75': 100*stats[2],
                             'unit': 'AP points (0-100)'}
        return evaluated
    metric.run_coco_eval = capture_eval
    if args.method != 'B0':
        from fsod_experiments.upstream_adapter import install
        config = json.loads((ROOT / 'configs/stage1.json').read_text())
        install(support_util, args.method, config, args.lam, args.mu, out,
                {'baseline_commit': record['baseline_commit'], 'weights': record['model_weights_hash'],
                 'image_size': 630, 'precision': 'official_fp32', 'implementation': record['implementation_hash']})
    # Preserve official defaults and graph parameters; evaluate all domain classes.
    sys.argv = ['main.py', '--json_path', str(paths['support']), '--test_json', str(paths['query']),
                '--test_img_dir', str(paths['query_images']), '--data_dir', str(paths['data_root']),
                '--model_version', 'dinov2_vitl14', '--feat_extractor_name', 'DINOV2',
                '--repo_or_dir', str(upstream / 'dinov2'), '--dinov2_checkpoint_dir', str(ROOT / 'checkpoints'),
                '--pred_json', str(out / 'predictions.json'), '--min_threshold', '0.01',
                '--diffusion_steps', '30', '--alp', '0.3', '--lamb', '0.5']
    started = time.perf_counter()
    try:
        runpy.run_path(str(upstream / 'main.py'), run_name='__main__')
        torch.cuda.synchronize()
        record['status'] = 'completed'
    except BaseException:
        record['status'] = 'failed'; record['error'] = traceback.format_exc(); raise
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        record['peak_gpu_memory_gb'] = torch.cuda.max_memory_allocated()/1024**3
        write_json(out / 'run.json', record)
    return 0


if __name__ == '__main__':sys.exit(main())
