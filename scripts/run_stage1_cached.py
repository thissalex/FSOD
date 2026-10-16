#!/usr/bin/env python3
"""Run development H1 and E2/E3 with fixed, parity-checked official proposals.

No official test label enters adaptation or hyperparameter selection. Default
lambda=10, mu=1 are prespecified pilot settings, not selected best parameters.
Original upstream code is verified and instrumented in memory, never rewritten.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fsod_experiments.adaptation import fit, normalize
from fsod_experiments.views import make_views
from fsod_experiments.records import digest, file_digest, write_json, paired_summary
from fsod_experiments.cached_detection import instrument,save_query,replay,assert_parity,BASELINE_COMMIT
from pixel_protection import protected_views


def evaluate(query_path,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(query_path));gt.dataset.setdefault('info',{})
        if predictions:dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
        evaluator=COCOeval(gt,dt,'bbox');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    return {'ap':float(100*evaluator.stats[0]),'ap50':float(100*evaluator.stats[1]),
            'ap75':float(100*evaluator.stats[2])}


def support_views(manifest,sam,model,transform,util,config,cache_identity):
    features=[];labels=[];diagnostics=[];names=sorted(manifest)
    started=time.perf_counter()
    for label,name in enumerate(names):
        for sample in manifest[name]:
            path=ROOT/'dataset'/sample['image']
            key=digest({'identity':cache_identity,'sample':sample,'image':file_digest(path),'config':config})
            destination=ROOT/'cache/stage1_support'/f'{key}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                with np.load(destination,allow_pickle=False) as saved:
                    x=torch.from_numpy(saved['features'].copy())
                    diagnostic=json.loads(str(saved['diagnostic']))
            else:
                image=Image.open(path).convert('RGB');bx,by,bw,bh=sample['bbox']
                views,meta,leaks,padding=protected_views(image,[bx,by,bx+bw,by+bh],config,
                                                        make_views,util.resize_with_aspect_ratio,transform)
                mask=util.extract_masks_from_support(sam,image,[bx,by,bx+bw,by+bh],'cuda')
                vectors=[];fixed=None
                for view in views+[image.copy()]:
                    dense=util.get_dinov2_features(model,transform,view,'cuda')
                    if fixed is None:
                        projected=util.resize_mask_to_features(mask,dense.shape[2:])
                        fixed=torch.from_numpy(projected).to('cuda')[None,None]
                        if fixed.sum()==0:raise RuntimeError(f'Empty support mask: {sample}')
                    vectors.append(((dense*fixed).sum((2,3))/fixed.sum())[0].float().cpu())
                x=torch.stack(vectors[:-1]);z=normalize(x.double())
                repeat=normalize(vectors[-1].double())
                diagnostic={'category':name,'instance_id':sample['instance_id'],'cache_key':key,
                            'cosine_drift':(1-(z[0]*z).sum(-1)).tolist(),
                            'repeat_noise':float(1-(z[0]*repeat).sum()),
                            'target_input_max_delta_by_view':leaks,'perturbable_fraction':meta[0]['perturbable_fraction'],
                            'interpolation_guard_pixels':padding}
                temp=destination.with_suffix('.tmp.npz')
                np.savez_compressed(temp,features=x.numpy(),diagnostic=json.dumps(diagnostic));temp.replace(destination)
            features.append(x);labels.append(label);diagnostics.append(diagnostic)
    return torch.stack(features),torch.tensor(labels),names,diagnostics,time.perf_counter()-started


def query_cache(query_path,image_root,model,transform,sam,util,metric,p,classes,upn,identity,output):
    from pycocotools.coco import COCO
    data=json.loads(query_path.read_text());image_ids=[i['id'] for i in data['images']]
    category_ids={c['name']:c['id'] for c in data['categories']}
    # Image contents, not only file names, participate in cache identity.
    key=digest({'identity':identity,'query':file_digest(query_path),
                'images':{i['id']:file_digest(image_root/i['file_name']) for i in data['images']}})
    directory=ROOT/'cache/stage1_query'/key;directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if not all((directory/f'{i}.npz').is_file() for i in image_ids):
            raise RuntimeError('Incomplete query cache')
        return directory,image_ids,category_ids,manifest
    # A partial cache is only a resume hint, never accepted as a completed run.
    captured=set()
    def capture(image_id,items):
        save_query(directory,image_id,items);captured.add(image_id)
    generator=instrument(metric.generate_coco_style_predictions_upn,capture)
    gt=COCO(str(query_path));started=time.perf_counter()
    with torch.inference_mode():
        predictions=generator(gt,str(image_root),sam,'DINOV2',model,transform,p.T.to('cuda'),classes,
                              upn,30,0.3,0.5,'cuda',0.01)
    # Official code can continue early when no proposals exist. Record an empty cache
    # for those images; all images are still present in COCO evaluation ground truth.
    for image_id in set(image_ids)-captured:save_query(directory,image_id,[])
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    replayed=replay(directory,image_ids,p,[category_ids[c] for c in classes],metric.graph_diffusion_ios)
    parity=assert_parity(predictions,replayed)
    manifest={'key':key,'identity':identity,'image_ids':image_ids,'query_sha256':file_digest(query_path),
              'parity':parity,'query_generation_seconds':elapsed,
              'live_baseline_metrics':evaluate(query_path,predictions),
              'empty_proposal_images':sorted(set(image_ids)-captured)}
    write_json(manifest_path,manifest);write_json(output/'query_cache_validation.json',manifest)
    return directory,image_ids,category_ids,manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['NEU-DET','clipart1k'],required=True)
    parser.add_argument('--seeds',type=int,default=3);parser.add_argument('--shots',type=int,nargs='+',default=[5,1])
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    out=ROOT/'results/resume'/args.domain;out.mkdir(parents=True,exist_ok=True)
    status={'status':'running','domain':args.domain,'baseline_commit':BASELINE_COMMIT,
            'configuration':'prespecified pilot lambda=10 mu=1, not parameter-selected',
            'near_duplicate_review':'exact RGB duplicates grouped; perceptual review pending'}
    write_json(out/'status.json',status)
    upstream=ROOT/'external/FSOD-VFM'
    baseline=json.loads((upstream/'source_manifest.json').read_text())
    for item in baseline['files']:
        body=(upstream/item['path']).read_bytes()
        if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=item['sha']:
            raise ValueError(f'Modified upstream: {item["path"]}')
    identity={'baseline_commit':BASELINE_COMMIT,'runner_sha256':file_digest(Path(__file__)),
              'capture_sha256':file_digest(ROOT/'fsod_experiments/cached_detection.py'),
              'math_sha256':file_digest(ROOT/'fsod_experiments/adaptation.py'),
              'views_sha256':file_digest(ROOT/'fsod_experiments/views.py'),
              'pixel_protection_sha256':file_digest(ROOT/'scripts/pixel_protection.py'),
              'weights':{p.name:file_digest(p) for p in sorted((ROOT/'checkpoints').glob('*')) if p.suffix in ['.pt','.pth']},
              'image_size':630,'precision':'official_fp32','proposal_threshold':0.01,'max_proposals':100,
              'diffusion_steps':30,'graph_alpha':0.3,'graph_decay':0.5}
    config=json.loads((ROOT/'configs/stage1.json').read_text())
    prepared=ROOT/'artifacts/development'/args.domain
    query_path=prepared/'query.json'
    protocol=json.loads((ROOT/'results/resume/development_protocol.json').read_text())
    domain_protocol=next(d for d in protocol if d['domain']==args.domain)
    if domain_protocol['query_sha256']!=file_digest(query_path):raise ValueError('Development query changed')
    os.chdir(upstream);sys.path.insert(0,str(upstream))
    import support_util as util
    import metric
    from model.dinov2 import load_dinov2_model
    from model.sam2 import load_sam2_components
    from chatrex.upn import UPNWrapper
    began=time.perf_counter()
    model,transform=load_dinov2_model('cuda','dinov2_vitl14',(630,630),str(ROOT/'external/dinov2'),
                                     str(ROOT/'checkpoints/dinov2_vitl14_pretrain.pth'))
    weights=torch.load(ROOT/'checkpoints/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True);del weights
    sam_model,sam,_=load_sam2_components('large','cuda')
    upn=UPNWrapper(str(ROOT/'checkpoints/upn_large.pth'))
    rows=[];all_diagnostics=[];cache=None;failed=[]
    for shot in args.shots:
        for seed in range(args.seeds):
            manifest_path=prepared/f'support_{shot}shot_seed{seed}.json';manifest=json.loads(manifest_path.read_text())
            expected=next(s for s in domain_protocol['support_manifests'] if s['shot']==shot and s['seed']==seed)
            if file_digest(manifest_path)!=expected['sha256']:raise ValueError('Support manifest changed')
            if any(len(v)!=shot for v in manifest.values()):raise ValueError('Wrong support count')
            print(f'{args.domain} {shot}-shot seed={seed}: extracting support views',flush=True)
            try:
                with torch.inference_mode():
                    x,labels,names,diagnostics,support_time=support_views(manifest,sam,model,transform,util,config,identity)
            except RuntimeError as exc:
                failed.append({'shot':shot,'seed':seed,'error':str(exc)})
                write_json(out/'failures.json',failed);raise
            all_diagnostics.append({'shot':shot,'seed':seed,'instances':diagnostics})
            write_json(out/'sensitivity.json',all_diagnostics)
            # Original image / original aggregation for baseline prototype; no views averaged.
            original_p=F.normalize(torch.stack([x[labels==c,0].mean(0) for c in range(len(names))]),dim=1)
            if cache is None:
                cache=query_cache(query_path,ROOT/'dataset/CDFSOD'/args.domain/'train',model,transform,sam,
                                  util,metric,original_p,names,upn,identity,out)
            directory,image_ids,category_ids,qmanifest=cache
            for method in (['B0','B1','B2','B5','Ours'] if shot==5 else ['B0','B1','Ours']):
                started=time.perf_counter();a=None;stats={}
                if method=='B0':p=original_p
                elif method=='B2':
                    # View indices: 0 original, 1 blur2, 2 blur5, 3 bright0.6, 4 bright1.4.
                    p=F.normalize(torch.stack([x[labels==c,2].mean(0) for c in range(len(names))]),dim=1)
                else:
                    matrix,p,_,stats=fit(x,labels,10 if method in ['B5','Ours'] else 0,
                                         1 if method=='Ours' else 0)
                    p=p.float();a=matrix if method in ['B5','Ours'] else None
                predictions=replay(directory,image_ids,p,[category_ids[c] for c in names],metric.graph_diffusion_ios,a)
                metrics=evaluate(query_path,predictions)
                row={'domain':args.domain,'shot':shot,'seed':seed,'method':method,'status':'completed',**metrics,
                     'support_manifest_sha256':file_digest(manifest_path),'query_manifest_sha256':file_digest(query_path),
                     'query_cache_key':qmanifest['key'],'lambda':10 if method in ['B5','Ours'] else 0,
                     'mu':1 if method=='Ours' else 0,'support_seconds_shared':support_time,
                     'replay_and_evaluation_seconds':time.perf_counter()-started,**stats}
                rows.append(row);write_json(out/'paired_results.json',rows)
                print(json.dumps(row),flush=True)
    summaries=[]
    for shot in args.shots:
        for control in ['B0','B1']:
            left={str(r['seed']):r['ap'] for r in rows if r['shot']==shot and r['method']==control}
            right={str(r['seed']):r['ap'] for r in rows if r['shot']==shot and r['method']=='Ours'}
            summaries.append({'shot':shot,'control':control,'treatment':'Ours',**paired_summary(left,right)})
    write_json(out/'paired_summary.json',summaries)
    status.update(status='completed',evaluations=len(rows),elapsed_seconds=time.perf_counter()-began,
                  peak_memory_gb=torch.cuda.max_memory_allocated()/1024**3,
                  identity=identity,query_cache_key=cache[3]['key'])
    write_json(out/'status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        # Preserve failures and prior partial evaluations; do not imply a complete matrix.
        import traceback
        parser_domain=next((sys.argv[i+1] for i,v in enumerate(sys.argv[:-1]) if v=='--domain'),'unknown')
        path=ROOT/'results/resume'/parser_domain/'status.json'
        data=json.loads(path.read_text()) if path.exists() else {}
        data.update(status='failed',error=traceback.format_exc());write_json(path,data)
        raise
