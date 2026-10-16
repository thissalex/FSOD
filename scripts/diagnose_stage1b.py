#!/usr/bin/env python3
"""Original development grid diagnostics, preserving every stage-one artifact."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/FSOD-VFM'),str(ROOT/'scripts')]
from fsod_experiments.adaptation import fit
from fsod_experiments.diagnostics import response_statistics,spectrum,distribution,aligned_prediction,ranking_changes,coco_metrics
from fsod_experiments.records import write_json,file_digest
from fsod_experiments.cached_detection import replay,assert_parity
from analyze_context_errors import iou_xywh
import metric


def load_batch(domain,seed,shot=5):
    batches=json.loads((ROOT/'results/resume'/domain/'sensitivity.json').read_text())
    batch=next(b for b in batches if b['shot']==shot and b['seed']==seed)
    names=sorted({i['category'] for i in batch['instances']});features=[];labels=[]
    for instance in batch['instances']:
        with np.load(ROOT/'cache/stage1_support'/f'{instance["cache_key"]}.npz',allow_pickle=False) as data:
            features.append(torch.from_numpy(data['features'].copy()));labels.append(names.index(instance['category']))
    return torch.stack(features),torch.tensor(labels),names,batch


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        writer.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['NEU-DET','clipart1k']);args=parser.parse_args()
    torch.set_num_threads(4);config=json.loads((ROOT/'configs/stage1b.json').read_text())
    began=time.perf_counter();transforms=[];predrows=[];seed_index=[];parities=[]
    out=ROOT/'results/stage1b'/(args.domain or 'combined');out.mkdir(parents=True,exist_ok=True)
    write_json(out/'diagnostic_status.json',{'status':'running','config_sha256':file_digest(ROOT/'configs/stage1b.json')})
    for domain in ([args.domain] if args.domain else config['domains']):
        query_path=ROOT/'artifacts/development_v2'/domain/'query.json';query=json.loads(query_path.read_text())
        validation=json.loads((ROOT/'results/resume'/domain/'query_cache_validation.json').read_text())
        revision=next(r for r in json.loads((ROOT/'results/stage1b/protocol_revision.json').read_text()) if r['domain']==domain)
        if file_digest(query_path)!=revision['query_sha256']:raise ValueError('Changed v2 query manifest')
        image_ids=[i['id'] for i in query['images']]
        directory=ROOT/'cache/stage1_query'/validation['key'];catnames={c['id']:c['name'] for c in query['categories']}
        catids={v:k for k,v in catnames.items()};gt=defaultdict(list)
        for ann in query['annotations']:
            if not ann.get('iscrowd',0):gt[ann['image_id']].append(ann)
        caches=[]
        for image_id in image_ids:
            with np.load(directory/f'{image_id}.npz',allow_pickle=False) as data:
                q=torch.from_numpy(data['features'].copy()).cuda();masks=torch.from_numpy(data['masks'].copy()).cuda();boxes=data['boxes'].copy()
            if not len(q):continue
            selected=[]
            if gt[image_id]:
                overlaps=iou_xywh([a['bbox'] for a in gt[image_id]],boxes)
                selected=[(int(overlaps[j].argmax()),a['category_id']) for j,a in enumerate(gt[image_id]) if overlaps[j].max()>=.5]
            caches.append((image_id,q,masks,boxes,selected))
        grid=json.loads((ROOT/'results/resume'/domain/'prespecified_grid.json').read_text())
        for seed in config['development_seeds']:
            x,y,names,batch=load_batch(domain,seed);ids=[catids[n] for n in names];labelmap={c:i for i,c in enumerate(ids)}
            e,pc,_=response_statistics(x,y);c=e@e.T;s=pc@pc.T;cs=spectrum(c);ss=spectrum(s)
            _,p,_,_=fit(x,y,0,0);p=p.float().cuda();baseline=[];old=[]
            for image_id,q,masks,boxes,selected in caches:
                q0,sim0,labels0,final0=aligned_prediction(q,p,masks,metric.graph_diffusion_ios)
                old.append((sim0,labels0,final0))
                baseline.extend({'image_id':image_id,'bbox':box.tolist(),'category_id':ids[int(label)],'score':float(score)} for box,label,score in zip(boxes,labels0,final0))
            baseap,baseclasses=coco_metrics(query_path,baseline)
            seed_index.append({'domain':domain,'seed':seed,'shot':5,'B1_ap_recomputed':baseap,'query_cache_key':validation['key'],
                'support_manifest_sha256':file_digest(ROOT/'artifacts/development'/domain/f'support_5shot_seed{seed}.json'),
                'support_cache_keys':[i['cache_key'] for i in batch['instances']],
                'source_grid':'results/resume/'+domain+'/prespecified_grid.json'})
            for lam in config['lambda_grid']:
                for mu in config['mu_grid'] if lam else [0]:
                    a,_,_,stats=fit(x,y,lam,mu);af=a.float().cuda() if lam else None
                    row={'domain':domain,'shot':5,'seed':seed,'lambda':lam,'mu':mu,'instances':len(x),'views':x.shape[1],'dimension':x.shape[2],
                        'rank_bound':min(x.shape[2],len(x)*(x.shape[1]-1)),
                        **{'C_'+k:v for k,v in cs.items()},**{'S_'+k:v for k,v in ss.items()},
                        'r_E':float(torch.linalg.norm(a@e)/torch.linalg.norm(e)) if e.norm()>0 else None,
                        'r_P':float(torch.linalg.norm((a-torch.eye(len(a)))@pc)/torch.linalg.norm(pc)) if pc.norm()>0 else None,
                        'solve_relative_residual':stats['solve_relative_residual'],
                        'identity_formula_residual':float(torch.linalg.norm((a-torch.eye(len(a)))@(torch.eye(len(a))+lam*c+mu*s)+lam*c)),
                        'finite':bool(torch.isfinite(a).all()),'support_min_norm':float(x.norm(dim=-1).min())}
                    zero_queries=0; nonfinite_queries=0;drift=[];maxdelta=[];flips=[];finaldelta=[];margindelta=[];rchanged=rtotal=0;predictions=[];querymin=[]
                    for (image_id,q,masks,boxes,selected),(sim0,lab0,fin0) in zip(caches,old):
                        qa,sims,labels,final=aligned_prediction(q,p,masks,metric.graph_diffusion_ios,af)
                        valid=(q.norm(dim=1)>=1e-12)&(qa.norm(dim=1)>=1e-12)
                        zero_queries+=int((~valid).sum());nonfinite_queries+=int((~torch.isfinite(qa).all(dim=1)).sum())
                        drift.extend((1-(F.normalize(q[valid],dim=1)*F.normalize(qa[valid],dim=1)).sum(1)).clamp(min=0).cpu().tolist())
                        maxdelta.extend((sims.max(1).values-sim0.max(1).values).abs().cpu().tolist())
                        flips.extend((labels!=lab0).cpu().tolist());finaldelta.extend((final-fin0).abs().cpu().tolist())
                        querymin.append(float((q if af is None else q@af.T).norm(dim=1).min()))
                        for idx,cat in selected:
                            k=labelmap[cat];other=[j for j in range(len(ids)) if j!=k]
                            margindelta.append(float((sims[idx,k]-sims[idx,other].max())-(sim0[idx,k]-sim0[idx,other].max())))
                        changed,total=ranking_changes(fin0.cpu().numpy(),final.cpu().numpy(),lab0.cpu().numpy(),labels.cpu().numpy());rchanged+=changed;rtotal+=total
                        predictions.extend({'image_id':image_id,'bbox':box.tolist(),'category_id':ids[int(label)],'score':float(score)} for box,label,score in zip(boxes,labels,final))
                    ap,classes=coco_metrics(query_path,predictions)
                    reference=next(r for r in grid if r['seed']==seed and r['lambda']==lam and r['mu']==mu)
                    if not revision['excluded_query_image_ids'] and abs(ap-reference['ap'])>1e-5:raise AssertionError(f'Grid replay AP mismatch {ap} vs {reference["ap"]}')
                    row['query_near_zero_count']=zero_queries;row['query_nonfinite_count']=nonfinite_queries
                    row.update(distribution(drift,'query_cosine_drift'));row['query_transformed_min_norm']=min(querymin);transforms.append(row)
                    summary={'domain':domain,'shot':5,'seed':seed,'lambda':lam,'mu':mu,'category':'ALL','candidate_count':len(flips),
                        'class_flip_rate':float(np.mean(flips)),'within_class_rank_pair_changes':rchanged,'within_class_rank_pairs':rtotal,
                        'within_class_rank_change_rate':rchanged/rtotal if rtotal else None,
                        **distribution(maxdelta,'max_similarity_abs_delta'),**distribution(finaldelta,'final_confidence_abs_delta'),
                        **distribution(margindelta,'correct_class_margin_delta'),'fixed_recalled_gt_count':len(margindelta),
                        'ap':ap,'B1_ap':baseap,'delta_ap':ap-baseap,'v1_ap':reference['ap'],'protocol':'v2'}
                    predrows.append(summary)
                    for cat,value in classes.items():
                        predrows.append({'domain':domain,'shot':5,'seed':seed,'lambda':lam,'mu':mu,'category':catnames[cat],
                                         'ap':value,'B1_ap':baseclasses[cat],'delta_ap':value-baseclasses[cat] if value is not None else None})
                    if lam==100 and mu==1:
                        # Independent evaluator agrees with production cache path, boxes aligned.
                        reference_preds=replay(directory,image_ids,p,ids,metric.graph_diffusion_ios,a)
                        parities.append({'domain':domain,'seed':seed,**assert_parity(predictions,reference_preds)})
                    write_csv(out/'transform_diagnostics.csv',transforms);write_csv(out/'prediction_diagnostics.csv',predrows)
                    print(json.dumps({k:summary[k] for k in ['domain','seed','lambda','mu','ap','delta_ap','class_flip_rate']}),flush=True)
    write_json(out/'seed_result_index.json',seed_index);write_json(out/'aligned_replay_parity.json',parities)
    write_json(out/'diagnostic_status.json',{'status':'completed','grid_rows':len(transforms),'seconds':time.perf_counter()-began,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'config_sha256':file_digest(ROOT/'configs/stage1b.json')})
if __name__=='__main__':main()
