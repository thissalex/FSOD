#!/usr/bin/env python3
"""Replay identity from a clean code copy and explicitly supplied local artifacts."""
import argparse
import json
from pathlib import Path
import sys
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fsod_experiments.adaptation import fit
from fsod_experiments.cached_detection import replay,assert_parity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.data_root.resolve()
    sys.path.insert(0,str(root/'external/FSOD-VFM'));import metric
    torch.set_num_threads(4);reports=[]
    for domain in ['NEU-DET','clipart1k']:
        batch=next(b for b in json.loads((root/'results/resume'/domain/'sensitivity.json').read_text()) if b['shot']==5 and b['seed']==0)
        names=sorted({i['category'] for i in batch['instances']});features=[];labels=[]
        for instance in batch['instances']:
            with np.load(root/'cache/stage1_support'/f'{instance["cache_key"]}.npz',allow_pickle=False) as data:features.append(torch.from_numpy(data['features'].copy()))
            labels.append(names.index(instance['category']))
        a,p,_,_=fit(torch.stack(features),torch.tensor(labels),0,0)
        validation=json.loads((root/'results/resume'/domain/'query_cache_validation.json').read_text())
        query=json.loads((root/'artifacts/development_v2'/domain/'query.json').read_text());cats={c['name']:c['id'] for c in query['categories']}
        image_ids=[i['id'] for i in query['images'][:2]];directory=root/'cache/stage1_query'/validation['key']
        left=replay(directory,image_ids,p.float(),[cats[n] for n in names],metric.graph_diffusion_ios)
        right=replay(directory,image_ids,p.float(),[cats[n] for n in names],metric.graph_diffusion_ios,a)
        reports.append({'domain':domain,'image_ids':image_ids,'code_root':str(ROOT),'data_root':str(root),**assert_parity(left,right)})
    args.output.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
