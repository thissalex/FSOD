"""Instrument pinned official inference before graph diffusion, without editing it.

Cache official FP32 region features, projected masks and boxes in their ORIGINAL
UPN order. Rebuild class groups and graph scores after each support adaptation.
An identity replay must agree with the live official prediction before reuse.
"""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
import torch
import torch.nn.functional as F


BASELINE_COMMIT='6da781bf02046fe6f70530e1d5483abd9aef6c9c'


def instrument(original, capture):
    """Insert one callback before the first image-level graph block, fail closed."""
    source=textwrap.dedent(inspect.getsource(original))
    module=ast.parse(source);function=module.body[0]
    loops=[n for n in function.body if isinstance(n,ast.For) and
           isinstance(n.target,ast.Name) and n.target.id=='img_dict']
    if len(loops)!=1:raise ValueError('Unexpected official image loop')
    loop=loops[0]
    graph=[i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and
           isinstance(n.test,ast.Name) and n.test.id=='img_results']
    if len(graph)!=2:raise ValueError('Unexpected official graph/postprocessing blocks')
    callback=ast.parse('_capture_query(img_id, img_results)').body[0]
    loop.body.insert(graph[0],callback)
    ast.fix_missing_locations(module)
    namespace={**original.__globals__,'_capture_query':capture}
    exec(compile(module,'<pinned-official-query-capture>','exec'),namespace)
    return namespace[function.name]


def save_query(directory, image_id, items):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/f'{image_id}.npz'
    if items:
        features=np.concatenate([i['feat'] for i in items],axis=0).astype(np.float32)
        masks=np.stack([i['masks_for_ios'] for i in items]).astype(np.uint8)
        boxes=np.array([i['bbox'] for i in items],dtype=np.float64)
    else:
        features=np.zeros((0,1024),dtype=np.float32)
        masks=np.zeros((0,1,1,1,1),dtype=np.uint8);boxes=np.zeros((0,4))
    temporary=destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,features=features,masks=masks,boxes=boxes)
    temporary.replace(destination)


def replay(directory, image_ids, prototypes, class_ids, graph, transform=None, device='cuda'):
    """Recompute matching and the ORIGINAL graph formula/order. No cached scores."""
    results=[]
    p=prototypes.to(device=device,dtype=torch.float32)
    if transform is not None:
        a=transform.to(device=device,dtype=torch.float32)
        p=F.normalize(p@a.T,dim=1,eps=1e-12)
    for image_id in image_ids:
        with np.load(Path(directory)/f'{image_id}.npz',allow_pickle=False) as data:
            if len(data['features'])==0:continue
            q=torch.from_numpy(data['features']).to(device)
            masks=torch.from_numpy(data['masks']).to(device)
            boxes=data['boxes'].copy()
        if transform is not None:q=F.normalize(q@a.T,dim=1,eps=1e-12)
        sims=q@p.T; scores,labels=sims.max(1)
        # Official code groups by actual predicted category IDs then remaps to indices.
        cats=[int(class_ids[i]) for i in labels.cpu().tolist()]
        unique=list(set(cats));mapping={c:i for i,c in enumerate(unique)}
        groups=torch.tensor([mapping[c] for c in cats],device=device)
        ios=graph(masks,groups,len(unique),max_iter=30,alpha=0.3)
        scores=scores*(1-ios).clamp(min=0).pow(0.5)
        items=[{'image_id':image_id,'category_id':cat,'bbox':box.tolist(),'score':float(score)}
               for cat,box,score in zip(cats,boxes,scores.cpu().tolist())]
        results.extend(sorted(items,key=lambda i:i['score'],reverse=True)[:100])
    return results


def assert_parity(original,replayed,tolerance=2e-5):
    def key(p):return (p['image_id'],tuple(p['bbox']))
    def grouped(predictions):
        groups={}
        for p in predictions:groups.setdefault(key(p),[]).append(p)
        return {k:sorted(v,key=lambda p:(p['category_id'],p['score'])) for k,v in groups.items()}
    left,right=grouped(original),grouped(replayed)
    if left.keys()!=right.keys():raise AssertionError('Cache replay changed candidate boxes')
    max_error=0.;count=0
    for k in left:
        if len(left[k])!=len(right[k]):raise AssertionError('Cache replay changed candidate count')
        for x,y in zip(left[k],right[k]):
            if x['category_id']!=y['category_id']:raise AssertionError('Cache replay changed category')
            max_error=max(max_error,abs(x['score']-y['score']));count+=1
    if max_error>tolerance:raise AssertionError(f'Cache score error: {max_error}')
    return {'detections':count,'max_score_error':max_error,'tolerance':tolerance,'passed':True}
