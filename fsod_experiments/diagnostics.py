"""S1B metrics on fixed candidate indices; no proposal/label-dependent adaptation."""
import contextlib
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
from fsod_experiments.adaptation import normalize


def response_statistics(x, labels):
    x=x.double();z=normalize(x)
    e=(z-z.mean(1,keepdim=True)).flatten(0,1).T/math.sqrt(x.shape[0]*x.shape[1])
    p=normalize(torch.stack([x[labels==c].mean((0,1)) for c in torch.unique(labels,sorted=True)]))
    pc=(p-p.mean(0)).T/math.sqrt(len(p))
    return e,pc,p


def spectrum(matrix):
    values=torch.linalg.eigvalsh(matrix).clamp(min=0).flip(0)
    trace=float(values.sum()); energy=values.cumsum(0)/trace if trace else torch.zeros_like(values)
    return {'trace':trace,'lambda_max':float(values[0]),
            'leading_eigenvalues':values[:10].tolist(),
            'cumulative_energy_top10':energy[:10].tolist(),
            'effective_rank':trace**2/float((values**2).sum()) if trace else None,
            'numerical_rank_relative_1e_10':int((values>values[0]*1e-10).sum()) if trace else 0}


def distribution(values, prefix):
    values=np.asarray(values,dtype=float).ravel()
    if len(values)==0:return {prefix+'_'+k:None for k in ['mean','p50','p95','max']}
    return {prefix+'_mean':float(values.mean()),prefix+'_p50':float(np.quantile(values,.5)),
            prefix+'_p95':float(np.quantile(values,.95)),prefix+'_max':float(values.max())}


def aligned_prediction(q, p, masks, graph, transform=None):
    """Return candidate-order scores, BEFORE final sorting; regroup every call."""
    if transform is not None:
        q=F.normalize(q@transform.T,dim=1,eps=1e-12)
        p=F.normalize(p@transform.T,dim=1,eps=1e-12)
    sims=q@p.T; maximum,labels=sims.max(1)
    # Graph uses only group equality, not numeric category IDs.
    groups,inverse=torch.unique(labels,return_inverse=True)
    ios=graph(masks,inverse,len(groups),max_iter=30,alpha=.3)
    final=maximum*(1-ios).clamp(min=0).pow(.5)
    return q,sims,labels,final


def ranking_changes(old_scores,new_scores,old_labels,new_labels):
    """Pair reversal/tie change only among candidates retaining the same class."""
    changed=total=0
    for label in np.unique(old_labels):
        idx=np.flatnonzero((old_labels==label)&(new_labels==label))
        if len(idx)<2:continue
        i,j=np.triu_indices(len(idx),1)
        a=np.sign(old_scores[idx[i]]-old_scores[idx[j]])
        b=np.sign(new_scores[idx[i]]-new_scores[idx[j]])
        changed+=int((a!=b).sum());total+=len(i)
    return changed,total


def coco_metrics(query_path,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(query_path));gt.dataset.setdefault('info',{})
        if predictions:dt=gt.loadRes(predictions)
        else:
            dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    per_class={}
    for i,cat in enumerate(ev.params.catIds):
        precision=ev.eval['precision'][:,:,i,0,-1];valid=precision[precision>=0]
        per_class[cat]=float(valid.mean()*100) if len(valid) else None
    return float(ev.stats[0]*100),per_class
