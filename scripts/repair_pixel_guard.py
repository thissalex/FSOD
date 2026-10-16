#!/usr/bin/env python3
"""Preserve initial experiments and repair only samples with interpolation leaks.

Query candidates/region caches and unaffected support features remain unchanged.
Re-evaluate affected support-set methods; original-image B0 must stay unchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.adaptation import fit,normalize
from fsod_experiments.records import digest,file_digest,write_json,paired_summary
from fsod_experiments.views import make_views
from fsod_experiments.cached_detection import replay
from pixel_protection import protected_views
from run_stage1_cached import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',required=True,choices=['NEU-DET','clipart1k'])
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    out=ROOT/'results/resume'/args.domain
    sensitivity=json.loads((out/'sensitivity.json').read_text())
    leaking={i['cache_key']:i for b in sensitivity for i in b['instances'] if max(i['target_input_max_delta_by_view'])>0}
    if not leaking:print('All input target pixels already protected');return
    backup=ROOT/'artifacts/resume_before_pixel_guard'/args.domain
    if backup.exists():raise ValueError('Pre-repair archive already exists; inspect before retrying')
    shutil.copytree(out,backup)
    config=json.loads((ROOT/'configs/stage1.json').read_text())
    upstream=ROOT/'external/FSOD-VFM';os.chdir(upstream);sys.path.insert(0,str(upstream))
    import support_util as util
    import metric
    from model.dinov2 import load_dinov2_model
    from model.sam2 import load_sam2_components
    model,transform=load_dinov2_model('cuda','dinov2_vitl14',(630,630),str(ROOT/'external/dinov2'),
                                     str(ROOT/'checkpoints/dinov2_vitl14_pretrain.pth'))
    state=torch.load(ROOT/'checkpoints/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    sam_model,sam,_=load_sam2_components('large','cuda')
    repaired={};report=[];affected=[]
    for batch in sensitivity:
        manifest=json.loads((ROOT/'artifacts/development'/args.domain/f'support_{batch["shot"]}shot_seed{batch["seed"]}.json').read_text())
        dirty=False
        for diagnostic in batch['instances']:
            old_key=diagnostic['cache_key']
            if old_key not in leaking:continue
            dirty=True
            if old_key in repaired:
                diagnostic.update(repaired[old_key]);continue
            sample=next(s for s in manifest[diagnostic['category']] if s['instance_id']==diagnostic['instance_id'])
            image=Image.open(ROOT/'dataset'/sample['image']).convert('RGB');bx,by,bw,bh=sample['bbox']
            views,metadata,leaks,padding=protected_views(image,[bx,by,bx+bw,by+bh],config,make_views,
                                                       util.resize_with_aspect_ratio,transform)
            with torch.inference_mode():
                mask=util.extract_masks_from_support(sam,image,[bx,by,bx+bw,by+bh],'cuda')
                vectors=[];fixed=None
                for view in views+[image.copy()]:
                    dense=util.get_dinov2_features(model,transform,view,'cuda')
                    if fixed is None:
                        fixed=torch.from_numpy(util.resize_mask_to_features(mask,dense.shape[2:])).to('cuda')[None,None]
                        if fixed.sum()==0:raise RuntimeError('Empty original mask during repair')
                    vectors.append(((dense*fixed).sum((2,3))/fixed.sum())[0].float().cpu())
            x=torch.stack(vectors[:-1]);z=normalize(x.double());repeat=normalize(vectors[-1].double())
            with np.load(ROOT/'cache/stage1_support'/f'{old_key}.npz',allow_pickle=False) as old:
                original=torch.from_numpy(old['features'][0].copy())
            if not torch.allclose(original,x[0],rtol=1e-5,atol=1e-5):
                raise RuntimeError('Original support feature changed; cannot isolate pixel-guard repair')
            # Retain the exact cached original feature for B0 consistency.
            x[0]=original
            new_key=digest({'original_key':old_key,'pixel_guard_version':1,'padding':padding,
                            'repair_script_sha256':file_digest(Path(__file__))})
            updated={**diagnostic,'cache_key':new_key,'cosine_drift':(1-(z[0]*z).sum(-1)).tolist(),
                     'repeat_noise':float(1-(z[0]*repeat).sum()),'target_input_max_delta_by_view':leaks,
                     'perturbable_fraction':metadata[0]['perturbable_fraction'],'interpolation_guard_pixels':padding}
            np.savez_compressed(ROOT/'cache/stage1_support'/f'{new_key}.npz',features=x.numpy(),diagnostic=json.dumps(updated))
            report.append({'instance_id':diagnostic['instance_id'],'old_key':old_key,'new_key':new_key,
                           'old_target_max_delta':max(diagnostic['target_input_max_delta_by_view']),
                           'new_target_max_delta':max(leaks),'guard_pixels':padding,
                           'original_feature_max_abs_error':float((original-vectors[0]).abs().max())})
            repaired[old_key]=updated;diagnostic.update(updated)
        if dirty:affected.append((batch['shot'],batch['seed']))
    write_json(out/'sensitivity.json',sensitivity)
    rows=json.loads((out/'paired_results.json').read_text())
    validation=json.loads((out/'query_cache_validation.json').read_text())
    query_path=ROOT/'artifacts/development'/args.domain/'query.json'
    if file_digest(query_path)!=validation['query_sha256']:raise ValueError('Development query changed')
    query=json.loads(query_path.read_text());category_ids={c['name']:c['id'] for c in query['categories']}
    for batch in sensitivity:
        if (batch['shot'],batch['seed']) not in affected:continue
        names=sorted({i['category'] for i in batch['instances']});features=[];labels=[]
        for instance in batch['instances']:
            with np.load(ROOT/'cache/stage1_support'/f'{instance["cache_key"]}.npz',allow_pickle=False) as saved:
                features.append(torch.from_numpy(saved['features'].copy()));labels.append(names.index(instance['category']))
        x=torch.stack(features);y=torch.tensor(labels)
        for row in rows:
            if (row['shot'],row['seed'])!=(batch['shot'],batch['seed']) or row['method']=='B0':continue
            a=None;stats={};method=row['method']
            if method=='B2':p=F.normalize(torch.stack([x[y==c,2].mean(0) for c in range(len(names))]),dim=1)
            else:
                a,p,_,stats=fit(x,y,10 if method in ['B5','Ours'] else 0,1 if method=='Ours' else 0)
                p=p.float();a=a if method in ['B5','Ours'] else None
            predictions=replay(ROOT/'cache/stage1_query'/validation['key'],validation['image_ids'],p,
                               [category_ids[n] for n in names],metric.graph_diffusion_ios,a)
            row.update(evaluate(query_path,predictions));row.update(stats)
            row['pixel_guard_repair']=True
            row['pre_repair_replay_and_evaluation_seconds']=row.get('replay_and_evaluation_seconds')
            row['replay_and_evaluation_seconds']=None  # repair replay was not independently profiled
            print(row['shot'],row['seed'],method,row['ap'],flush=True)
    write_json(out/'paired_results.json',rows)
    summaries=[]
    for shot in [5,1]:
        for control in ['B0','B1']:
            left={str(r['seed']):r['ap'] for r in rows if r['shot']==shot and r['method']==control}
            right={str(r['seed']):r['ap'] for r in rows if r['shot']==shot and r['method']=='Ours'}
            summaries.append({'shot':shot,'control':control,'treatment':'Ours',**paired_summary(left,right)})
    write_json(out/'paired_summary.json',summaries)
    write_json(out/'pixel_guard_repair.json',{'status':'completed','affected_support_sets':affected,
                                           'repairs':report,'selection_rule':'Pixel equality only; no AP/feature-driven guard choice',
                                           'query_cache_unchanged':validation['key']})
    status=json.loads((out/'status.json').read_text());status['pixel_guard_repair']='completed'
    status['core_source_after_repair']=file_digest(ROOT/'scripts/run_stage1_cached.py')
    write_json(out/'status.json',status)


if __name__=='__main__':main()
