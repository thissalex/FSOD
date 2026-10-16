#!/usr/bin/env python3
"""Validate completed cells, fixed inputs, numeric-source identities and snapshots."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import file_digest,write_json


def main():
    out=ROOT/'results/stage1b';summary=json.loads((out/'summary.json').read_text())
    if summary['decision']!='stop_current_closed_form':raise ValueError('Review conditional branch before validation')
    statuses=['NEU-DET/diagnostic_status.json','clipart1k/diagnostic_status.json','proposal_status.json','controls_status.json','nonidentity_status.json','class_margin_status.json']
    for f in statuses:
        if json.loads((out/f).read_text())['status']!='completed':raise ValueError(f'Incomplete: {f}')
    numeric=json.loads((out/'executed_numeric_sources.json').read_text())
    for f,expected in numeric.items():
        if file_digest(ROOT/f)!=expected:raise AssertionError(f'Executed numeric source changed: {f}')
    if subprocess.check_output(['git','diff','d45f353c11b43f47dac5749fa7620947ca7b50e3','--','results/resume','stage1_resume_results.md'],cwd=ROOT):raise AssertionError('Stage-one snapshot changed')
    manifest=json.loads((out/'repro_manifest.json').read_text())
    for f,expected in manifest['source_sha256'].items():
        if file_digest(ROOT/f)!=expected:raise AssertionError(f'Snapshot source changed: {f}')
    for revision in json.loads((out/'protocol_revision.json').read_text()):
        data=json.loads((ROOT/revision['query_relative_path']).read_text());ids={i['id'] for i in data['images']}
        if file_digest(ROOT/revision['query_relative_path'])!=revision['query_sha256']:raise ValueError('Query changed')
        if ids&set(revision['excluded_query_image_ids']):raise ValueError('Exclusion not applied')
        support=json.loads((ROOT/'artifacts/development'/revision['domain']/'support_pool.json').read_text())
        if ids&{i['id'] for i in support['images']}:raise ValueError('Support/query image overlap')
        for row in json.loads((out/'nonidentity_parity.json').read_text()):
            if row['domain']==revision['domain'] and (not row['passed'] or not set(row['image_ids'])<=ids):raise ValueError('Invalid nonidentity parity')
        original=next(r for r in json.loads((ROOT/'results/resume/development_protocol.json').read_text()) if r['domain']==revision['domain'])
        if file_digest(ROOT/'artifacts/development'/revision['domain']/'support_pool.json')!=original['support_pool_sha256']:raise ValueError('Support pool changed')
        for item in original['support_manifests']:
            if file_digest(ROOT/'artifacts/development'/revision['domain']/f'support_{item["shot"]}shot_seed{item["seed"]}.json')!=item['sha256']:raise ValueError('Support draws changed')
    transform=list(csv.DictReader((ROOT/'transform_diagnostics.csv').open()));pred=list(csv.DictReader((ROOT/'prediction_diagnostics.csv').open()))
    if len(transform)!=60 or len([r for r in pred if r['category']=='ALL'])!=60:raise ValueError('Missing diagnostic rows')
    if any(r['finite']!='True' or int(r['query_nonfinite_count']) for r in transform):raise ValueError('Nonfinite transform/query')
    if any(int(r['seed'])>2 for r in transform):raise ValueError('Reserved seed evaluated')
    for domain in ['NEU-DET','clipart1k']:
        for r in json.loads((out/domain/'aligned_replay_parity.json').read_text()):
            if not r['passed']:raise ValueError('Aligned parity failed')
    clean=json.loads((out/'clean_replay.json').read_text())
    if len(clean)!=2 or not all(r['passed'] for r in clean):raise ValueError('Clean replay failed')
    pytest=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True,check=True)
    pip=subprocess.run([sys.executable,'-m','pip','check'],cwd=ROOT,capture_output=True,text=True,check=True)
    (ROOT/'logs/stage1b_tests.log').write_text(pytest.stdout+pytest.stderr);(ROOT/'logs/stage1b_pip_check.log').write_text(pip.stdout+pip.stderr)
    # A conservative allocation-window bound includes idle/CPU intervals, not just kernels.
    start=json.loads((out/'campaign_start.json').read_text())['unix_seconds']
    resource={'accounting':'per-process GPU-assigned wall time where recorded; two-device elapsed-window upper bound includes idle time and CPU work, not kernel-active time',
        'completed_process_seconds':{f:json.loads((out/f).read_text()).get('seconds') for f in statuses},
        'earlier_replaced_run_seconds':{'UPN_initial':89.42399442498572,'nonidentity_initial':33.41959425085224,'proposal_v2_first_cached_reanalysis':20.06614084006287},
        'initial_diagnostic_aborted':'SIGTERM after near-duplicate discovery; partial output kept in artifacts/stage1b/v1_partial, not used in v2 decision',
        'two_gpu_elapsed_window_upper_bound_hours':2*(time.time()-start)/3600 if start else None,
        'cap_hours':12,'peak_allocated_gib_by_completed_process':{f:json.loads((out/f).read_text()).get('peak_gpu_allocated_gib') for f in statuses}}
    if resource['two_gpu_elapsed_window_upper_bound_hours'] is None or resource['two_gpu_elapsed_window_upper_bound_hours']>=12:raise ValueError('Resource bound unavailable or exceeded')
    logs={p.name:{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in sorted((ROOT/'logs').glob('stage1b*')) if p.is_file()}
    write_json(out/'validation.json',{'status':'passed','pytest':pytest.stdout.strip(),'pip_check':pip.stdout.strip(),
        'grid_rows':60,'core_controls':48,'core_recomputed':24,'v1_snapshot_unchanged':True,
        'source_and_support_digests_verified':True,'nonidentity_and_aligned_parity':True,'clean_replay':True,
        'heldout_method_evaluations':0,'resource':resource,'local_log_index':logs})
    print(pytest.stdout.strip());print(pip.stdout.strip());print('GPU allocation window upper bound (hours):',resource['two_gpu_elapsed_window_upper_bound_hours'])
if __name__=='__main__':main()
