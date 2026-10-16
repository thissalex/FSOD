#!/usr/bin/env python3
"""Evaluate only the preregistered 5-shot development lambda/mu grid.

Collapse lambda=0 identity duplicates and reuse mathematically identical pilot
evaluations. No model inference, query-set statistics or official test selection.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fsod_experiments.adaptation import fit
from fsod_experiments.cached_detection import replay
from fsod_experiments.records import write_json,paired_summary,file_digest

spec=importlib.util.spec_from_file_location('stage1_runner',ROOT/'scripts/run_stage1_cached.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',required=True,choices=['NEU-DET','clipart1k'])
    parser.add_argument('--seeds',type=int,nargs='+',help='Recompute only these seeds, preserving other completed grid rows')
    args=parser.parse_args();torch.set_num_threads(4)
    directory=ROOT/'results/resume'/args.domain
    status=json.loads((directory/'status.json').read_text())
    if status['status']!='completed':raise ValueError('Complete core experiment before running the grid')
    write_json(directory/'grid_status.json',{'status':'running','recomputed_seeds':args.seeds})
    config=json.loads((ROOT/'configs/stage1.json').read_text())
    validation=json.loads((directory/'query_cache_validation.json').read_text())
    if not validation['parity']['passed']:raise ValueError('Cache parity failed')
    query_path=ROOT/'artifacts/development'/args.domain/'query.json'
    if file_digest(query_path)!=validation['query_sha256']:raise ValueError('Query manifest changed')
    query=json.loads(query_path.read_text());name_to_id={c['name']:c['id'] for c in query['categories']}
    pilots=json.loads((directory/'paired_results.json').read_text())
    sensitivity=json.loads((directory/'sensitivity.json').read_text())
    sys.path.insert(0,str(ROOT/'external/FSOD-VFM'));import metric
    rows=[]
    if args.seeds and (directory/'prespecified_grid.json').exists():
        rows=[r for r in json.loads((directory/'prespecified_grid.json').read_text()) if r['seed'] not in args.seeds]
    for batch in sensitivity:
        if batch['shot']!=5 or (args.seeds and batch['seed'] not in args.seeds):continue
        names=sorted({i['category'] for i in batch['instances']});features=[];labels=[]
        for instance in batch['instances']:
            with np.load(ROOT/'cache/stage1_support'/f'{instance["cache_key"]}.npz',allow_pickle=False) as data:
                features.append(torch.from_numpy(data['features'].copy()))
                labels.append(names.index(instance['category']))
        x=torch.stack(features);y=torch.tensor(labels)
        seed=batch['seed']
        for lam in config['lambda_grid']:
            for mu in (config['mu_grid'] if lam else [0]):
                began=time.perf_counter()
                method='B1' if lam==0 else 'B5' if (lam==10 and mu==0) else 'Ours' if (lam==10 and mu==1) else None
                reuse=next((r for r in pilots if r['shot']==5 and r['seed']==seed and r['method']==method),None)
                if reuse:
                    metrics={k:reuse[k] for k in ['ap','ap50','ap75']}
                else:
                    a,p,_,_=fit(x,y,lam,mu)
                    predictions=replay(ROOT/'cache/stage1_query'/validation['key'],validation['image_ids'],
                                       p.float(),[name_to_id[n] for n in names],metric.graph_diffusion_ios,a if lam else None)
                    metrics=runner.evaluate(query_path,predictions)
                rows.append({'domain':args.domain,'shot':5,'seed':seed,'lambda':lam,'mu':mu,**metrics,
                             'reused_pilot':bool(reuse),'query_cache_key':validation['key'],
                             'query_sha256':validation['query_sha256'],'seconds':time.perf_counter()-began})
                write_json(directory/'prespecified_grid.json',rows)
                print(json.dumps(rows[-1]),flush=True)
    write_json(directory/'grid_status.json',{'status':'completed','evaluations':len(rows),
                                           'source_sha256':file_digest(Path(__file__)),
                                           'scope':'prespecified development grid; lambda=0 duplicates collapsed'})


if __name__=='__main__':main()
