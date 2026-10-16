#!/usr/bin/env python3
"""Describe H2 within domain/category using fixed maximum-IoU query candidates.

Uses development labels only. Candidate recall is independent of adaptation;
association across three support seeds is descriptive, never a causal estimate.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]


def iou_xywh(left,right):
    left=np.asarray(left,dtype=float).reshape(-1,4);right=np.asarray(right,dtype=float).reshape(-1,4)
    l1=left[:,:2];l2=l1+left[:,2:];r1=right[:,:2];r2=r1+right[:,2:]
    overlap=np.maximum(np.minimum(l2[:,None],r2[None])-np.maximum(l1[:,None],r1[None]),0).prod(-1)
    return overlap/np.maximum(left[:,2:].prod(-1)[:,None]+right[:,2:].prod(-1)[None]-overlap,1e-12)


def main():
    reports=[]
    for domain in ['NEU-DET','clipart1k']:
        out=ROOT/'results/resume'/domain
        if not (out/'query_cache_validation.json').exists() or not (out/'sensitivity.json').exists():continue
        cache_manifest=json.loads((out/'query_cache_validation.json').read_text())
        cache=ROOT/'cache/stage1_query'/cache_manifest['key']
        query=json.loads((ROOT/'artifacts/development'/domain/'query.json').read_text())
        categories={c['id']:c['name'] for c in query['categories']}
        names=sorted(categories.values());name_to_label={name:i for i,name in enumerate(names)}
        by_image=defaultdict(list)
        for ann in query['annotations']:
            if not ann.get('iscrowd',0):by_image[ann['image_id']].append(ann)
        fixed=[];total=0;recalled=0
        for image in query['images']:
            gt=by_image[image['id']];total+=len(gt)
            if not gt:continue
            with np.load(cache/f'{image["id"]}.npz',allow_pickle=False) as data:
                if not len(data['boxes']):continue
                overlap=iou_xywh([a['bbox'] for a in gt],data['boxes'])
                selected=overlap.argmax(axis=1);valid=overlap.max(axis=1)>=0.5
                recalled+=int(valid.sum())
                for j in np.flatnonzero(valid):
                    ann=gt[j]
                    fixed.append((categories[ann['category_id']],data['features'][selected[j]].copy()))
        observations=[]
        for batch in json.loads((out/'sensitivity.json').read_text()):
            if batch['shot']!=5:continue
            sums=defaultdict(list);drifts=defaultdict(list)
            for instance in batch['instances']:
                with np.load(ROOT/'cache/stage1_support'/f'{instance["cache_key"]}.npz',allow_pickle=False) as data:
                    sums[instance['category']].append(data['features'][0].copy())
                drifts[instance['category']].append(float(np.mean(instance['cosine_drift'][1:])))
            prototypes=np.stack([np.mean(sums[n],axis=0) for n in names])
            prototypes/=np.linalg.norm(prototypes,axis=1,keepdims=True)
            errors=defaultdict(list);margins=defaultdict(list)
            for name,feature in fixed:
                sims=feature@prototypes.T;label=name_to_label[name]
                errors[name].append(int(sims.argmax()!=label))
                margins[name].append(float(sims[label]-np.max(np.delete(sims,label))))
            for name in names:
                observations.append({'category':name,'support_seed':batch['seed'],
                                     'support_mean_drift':float(np.mean(drifts[name])),
                                     'query_max_iou_recalled_instances':len(errors[name]),
                                     'classification_error_rate':float(np.mean(errors[name])) if errors[name] else None,
                                     'mean_class_margin':float(np.mean(margins[name])) if margins[name] else None})
        # Demean within each category before the pooled descriptive correlation.
        centered=[]
        for name in names:
            rows=[o for o in observations if o['category']==name and o['classification_error_rate'] is not None]
            if len(rows)<2:continue
            values=np.array([[r['support_mean_drift'],r['classification_error_rate'],r['mean_class_margin']] for r in rows])
            centered.extend(values-values.mean(0))
        associations={}
        if centered:
            centered=np.array(centered)
            for column,key in [(1,'drift_vs_error_rate'),(2,'drift_vs_margin')]:
                if np.ptp(centered[:,0])>0 and np.ptp(centered[:,column])>0:
                    associations[key]=float(spearmanr(centered[:,0],centered[:,column]).statistic)
        report={'domain':domain,'query_cache_key':cache_manifest['key'],
                'gt_instances':total,'fixed_candidate_recalled_at_iou50':recalled,
                'candidate_recall50':recalled/total,'observations':observations,
                'within_category_demeaned_spearman':associations,
                'interpretation':'Descriptive only: three correlated support-set runs; no significance or causal claim'}
        reports.append(report)
        print(domain,'candidate recall50',report['candidate_recall50'],associations,flush=True)
    (ROOT/'results/resume/context_error_diagnostics.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
