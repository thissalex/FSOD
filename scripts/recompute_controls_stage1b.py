#!/usr/bin/env python3
"""Recompute all prior method controls on the revised development query split.

Unchanged support draws and candidate/mask/feature caches. Nothing is overwritten
in results/resume. Includes prior 1-shot controls only to repair protocol records,
not to select a new 1-shot method or expand the experiment.
"""
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/FSOD-VFM'),str(ROOT/'scripts')]
from fsod_experiments.adaptation import fit
from fsod_experiments.cached_detection import replay
from fsod_experiments.records import write_json,file_digest
from fsod_experiments.diagnostics import coco_metrics
from diagnose_stage1b import load_batch
import metric


def main():
    torch.set_num_threads(4);started=time.perf_counter();rows=[]
    for revision in json.loads((ROOT/'results/stage1b/protocol_revision.json').read_text()):
        domain=revision['domain'];old=json.loads((ROOT/'results/resume'/domain/'paired_results.json').read_text())
        query=ROOT/revision['query_relative_path'];data=json.loads(query.read_text());catids={c['name']:c['id'] for c in data['categories']}
        ids=[i['id'] for i in data['images']]
        validation=json.loads((ROOT/'results/resume'/domain/'query_cache_validation.json').read_text())
        for row in old:
            revised={**row,'protocol':'v2','query_manifest_sha256':file_digest(query),'reused_v1':not bool(revision['excluded_query_image_ids'])}
            if revision['excluded_query_image_ids']:
                x,y,names,_=load_batch(domain,row['seed'],row['shot']);method=row['method'];a=None
                if method=='B0':p=F.normalize(torch.stack([x[y==c,0].mean(0) for c in range(len(names))]),dim=1)
                elif method=='B2':p=F.normalize(torch.stack([x[y==c,2].mean(0) for c in range(len(names))]),dim=1)
                else:
                    matrix,p,_,_=fit(x,y,row['lambda'],row['mu']);p=p.float()
                    if row['lambda']:a=matrix
                begin=time.perf_counter()
                predictions=replay(ROOT/'cache/stage1_query'/validation['key'],ids,p,[catids[n] for n in names],metric.graph_diffusion_ios,a)
                # Preserve AP50/AP75 via the same official COCO evaluator used in stage one.
                from run_stage1_cached import evaluate
                revised.update(evaluate(query,predictions));revised['v1_ap']=row['ap']
                revised['v2_replay_seconds']=time.perf_counter()-begin
                revised['replay_and_evaluation_seconds']=None
            rows.append(revised);write_json(ROOT/'results/stage1b/controls_v2.json',rows)
            print(domain,row['shot'],row['seed'],row['method'],revised['ap'],flush=True)
    write_json(ROOT/'results/stage1b/controls_status.json',{'status':'completed','rows':len(rows),'seconds':time.perf_counter()-started,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
if __name__=='__main__':main()
