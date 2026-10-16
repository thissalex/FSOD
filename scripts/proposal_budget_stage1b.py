#!/usr/bin/env python3
"""UPN-only nested budget recall; never extract additional SAM/DINO features."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/FSOD-VFM'),str(ROOT/'scripts')]
from fsod_experiments.records import write_json,file_digest,write_csv
from analyze_context_errors import iou_xywh


def clipped_prefix(boxes,budget,width,height):
    result=[]
    for x1,y1,x2,y2 in boxes[:budget]:
        x1,x2=np.clip([x1,x2],0,width);y1,y2=np.clip([y1,y2],0,height)
        if x2>x1 and y2>y1:result.append([x1,y1,x2-x1,y2-y1])
    return np.asarray(result,float).reshape(-1,4)


def main():
    torch.set_num_threads(4);config=json.loads((ROOT/'configs/stage1b.json').read_text());started=time.perf_counter()
    os.chdir(ROOT/'external/FSOD-VFM')
    from chatrex.upn import UPNWrapper
    upn=UPNWrapper(str(ROOT/'checkpoints/upn_large.pth'));rows=[];summaries=[]
    out=ROOT/'results/stage1b';write_json(out/'proposal_status.json',{'status':'running'})
    for domain in config['domains']:
        query_path=ROOT/'artifacts/development_v2'/domain/'query.json';query=json.loads(query_path.read_text())
        old=json.loads((ROOT/'results/resume'/domain/'query_cache_validation.json').read_text())
        cache=ROOT/'artifacts/stage1b/proposals'/domain;cache.mkdir(parents=True,exist_ok=True)
        cats={c['id']:c['name'] for c in query['categories']};annotations=defaultdict(list)
        for ann in query['annotations']:
            if not ann.get('iscrowd',0):annotations[ann['image_id']].append(ann)
        counts=defaultdict(list);hit_records=[];times=[];parity=0;uncovered=[]
        for index,im in enumerate(query['images']):
            dest=cache/f'{im["id"]}.npz';path=ROOT/'dataset/CDFSOD'/domain/'train'/im['file_name']
            if dest.exists():
                with np.load(dest,allow_pickle=False) as data:boxes=data['boxes'].copy();seconds=float(data['seconds'])
            else:
                with Image.open(path) as image:
                    image=image.convert('RGB');torch.cuda.synchronize();begin=time.perf_counter()
                    raw=upn.inference(image,config['proposal_prompt']);filtered=upn.filter(raw,min_score=config['proposal_min_score'],nms_value=config['proposal_nms'])
                    pairs=sorted(zip(filtered['scores'][0],filtered['original_xyxy_boxes'][0]),key=lambda x:x[0],reverse=True) if filtered['original_xyxy_boxes'] else []
                    boxes=np.array([b for _,b in pairs[:500]],dtype=float).reshape(-1,4)
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                np.savez_compressed(dest,boxes=boxes,seconds=seconds)
            times.append(seconds);gt=annotations[im['id']];gtboxes=[a['bbox'] for a in gt]
            for budget in config['proposal_budgets']:
                b=clipped_prefix(boxes,budget,im['width'],im['height']);counts[budget].append(len(b))
                if budget==100:
                    with np.load(ROOT/'cache/stage1_query'/old['key']/f'{im["id"]}.npz',allow_pickle=False) as original:
                        if b.shape!=original['boxes'].shape or not np.array_equal(b,original['boxes']):raise AssertionError(f'UPN top100 differs from baseline: {domain}/{im["id"]}')
                        parity+=1
                overlaps=iou_xywh(gtboxes,b) if len(b) else np.zeros((len(gt),0))
                best=overlaps.max(1) if len(b) else np.zeros(len(gt))
                for a,value in zip(gt,best):
                    area=a.get('area',a['bbox'][2]*a['bbox'][3]);size='small' if area<32**2 else 'medium' if area<96**2 else 'large'
                    hit_records.append((budget,cats[a['category_id']],size,float(value)))
                    if budget==500 and value<.5 and len(uncovered)<20:uncovered.append({'image_id':im['id'],'file_name':im['file_name'],'annotation_id':a['id'],'category':cats[a['category_id']],'bbox':a['bbox'],'best_iou':float(value)})
            if index%25==0:print(domain,index+1,len(query['images']),flush=True)
        groups=[('ALL','ALL')]
        if domain=='NEU-DET':groups += [(n,'ALL') for n in cats.values()]+[('ALL',s) for s in ['small','medium','large']]
        for budget in config['proposal_budgets']:
            for cat,size in groups:
                selected=[r[3] for r in hit_records if r[0]==budget and (cat=='ALL' or r[1]==cat) and (size=='ALL' or r[2]==size)]
                for threshold in config['proposal_iou']:
                    rows.append({'domain':domain,'budget':budget,'iou_threshold':threshold,'category':cat,'size':size,'gt_count':len(selected),
                        'recalled':sum(v>=threshold for v in selected),'recall':float(np.mean(np.array(selected)>=threshold)) if selected else None,
                        'actual_count_min':min(counts[budget]),'actual_count_mean':float(np.mean(counts[budget])),'actual_count_p50':float(np.median(counts[budget])),
                        'actual_count_max':max(counts[budget]),'generation_seconds_total':sum(times),'query_images':len(times)})
        summaries.append({'domain':domain,'query_sha256':file_digest(query_path),'top100_exact_box_parity_images':parity,
            'generation_seconds':sum(times),'actual_count_histograms':{str(b):{str(n):counts[b].count(n) for n in sorted(set(counts[b]))} for b in counts},
            'uncovered_at_500_iou50_examples':uncovered})
        write_csv(ROOT/'proposal_recall_budget.csv',rows);write_json(out/'proposal_summary.json',summaries)
    write_json(out/'proposal_status.json',{'status':'completed','seconds':time.perf_counter()-started,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'config_sha256':file_digest(ROOT/'configs/stage1b.json')})
if __name__=='__main__':main()
