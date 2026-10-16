#!/usr/bin/env python3
"""Within-class descriptive errors on fixed max-IoU recalled GT candidates.

One GT may share a candidate with another GT. These rows are descriptions of a
support-set run, never independent-sample significance tests.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from fsod_experiments.adaptation import fit
from fsod_experiments.records import write_json
from analyze_context_errors import iou_xywh
from diagnose_stage1b import load_batch,write_csv


def main():
    torch.set_num_threads(4);config=json.loads((ROOT/'configs/stage1b.json').read_text());rows=[]
    for domain in config['domains']:
        query=json.loads((ROOT/'artifacts/development_v2'/domain/'query.json').read_text())
        validation=json.loads((ROOT/'results/resume'/domain/'query_cache_validation.json').read_text());gt=defaultdict(list)
        for a in query['annotations']:
            if not a.get('iscrowd',0):gt[a['image_id']].append(a)
        catnames={c['id']:c['name'] for c in query['categories']};fixed=[];true=[]
        for im in query['images']:
            if not gt[im['id']]:continue
            with np.load(ROOT/'cache/stage1_query'/validation['key']/f'{im["id"]}.npz',allow_pickle=False) as data:
                if not len(data['boxes']):continue
                overlap=iou_xywh([a['bbox'] for a in gt[im['id']]],data['boxes'])
                for j,a in enumerate(gt[im['id']]):
                    if overlap[j].max()>=.5:fixed.append(data['features'][overlap[j].argmax()].copy());true.append(catnames[a['category_id']])
        q=torch.from_numpy(np.stack(fixed)).cuda()
        for seed in config['development_seeds']:
            x,y,names,_=load_batch(domain,seed);truth=torch.tensor([names.index(t) for t in true],device='cuda')
            _,p,_,_=fit(x,y,0,0);p=p.float().cuda();base=q@p.T
            idx=torch.arange(len(q),device='cuda')
            others=base.clone();others[idx,truth]=-torch.inf;base_margin=base[idx,truth]-others.max(1).values
            for lam in config['lambda_grid']:
                for mu in config['mu_grid'] if lam else [0]:
                    a,_,_,_=fit(x,y,lam,mu);a=a.float().cuda()
                    sims=F.normalize(q@a.T,dim=1,eps=1e-12)@F.normalize(p@a.T,dim=1,eps=1e-12).T if lam else base
                    others=sims.clone();others[idx,truth]=-torch.inf;margin=sims[idx,truth]-others.max(1).values
                    for k,name in enumerate(names):
                        selected=truth==k;n=int(selected.sum())
                        rows.append({'domain':domain,'shot':5,'seed':seed,'lambda':lam,'mu':mu,'category':name,'fixed_recalled_gt_count':n,
                            'B1_correct_class_margin_mean':float(base_margin[selected].mean()) if n else None,
                            'adapted_correct_class_margin_mean':float(margin[selected].mean()) if n else None,
                            'correct_class_margin_delta_mean':float((margin-base_margin)[selected].mean()) if n else None,
                            'B1_classification_error_rate':float((base.argmax(1)[selected]!=truth[selected]).float().mean()) if n else None,
                            'adapted_classification_error_rate':float((sims.argmax(1)[selected]!=truth[selected]).float().mean()) if n else None,
                            'GT_candidate_class_flip_rate':float((sims.argmax(1)[selected]!=base.argmax(1)[selected]).float().mean()) if n else None})
        write_csv(ROOT/'results/stage1b/class_margin_diagnostics.csv',rows)
    write_json(ROOT/'results/stage1b/class_margin_status.json',{'status':'completed','rows':len(rows),'matching':'fixed maximum-IoU original candidate per noncrowd GT at IoU>=0.5; descriptive only'})
if __name__=='__main__':main()
