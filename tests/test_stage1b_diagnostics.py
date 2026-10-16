import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fsod_experiments.adaptation import fit
from fsod_experiments.diagnostics import response_statistics,spectrum,aligned_prediction,ranking_changes


def test_suppression_and_discriminative_preservation_are_measured():
    g=torch.Generator().manual_seed(18)
    x=torch.randn(6,5,8,generator=g);labels=torch.tensor([0,0,0,1,1,1])
    e,pc,_=response_statistics(x,labels);a,_,_,_=fit(x,labels,100,1)
    c=e@e.T;s=pc@pc.T;eye=torch.eye(8,dtype=torch.double);h=eye+100*c+s
    assert torch.linalg.norm(a@e)<torch.linalg.norm(e)
    assert torch.allclose((a-eye)@h,-100*c,atol=1e-12)
    assert spectrum(c)['numerical_rank_relative_1e_10']<=min(8,len(x)*4)
    assert spectrum(torch.zeros(4,4))['effective_rank'] is None
    assert torch.allclose(fit(x,labels,0,10)[0],eye,atol=1e-14)


def test_nonidentity_changes_classes_and_regroups_graph():
    q=torch.tensor([[.6,.8],[.8,.6]])
    p=torch.eye(2);a=torch.diag(torch.tensor([.1,1.]))
    seen=[]
    def graph(masks,labels,class_num,**kwargs):
        seen.append((labels.tolist(),class_num))
        return torch.zeros(len(labels))
    base=aligned_prediction(q,p,torch.ones(2,1,1),graph)
    adapted=aligned_prediction(q,p,torch.ones(2,1,1),graph,a)
    assert base[2].tolist()==[1,0]
    assert adapted[2].tolist()==[1,1]
    assert seen[0][1]==2 and seen[1][1]==1
    expected=torch.nn.functional.normalize(q@a.T,dim=1)@torch.nn.functional.normalize(p@a.T,dim=1).T
    assert torch.allclose(adapted[1],expected)
    assert not torch.allclose(base[3],adapted[3])


def test_ranking_excludes_class_switches_and_counts_ties():
    changed,total=ranking_changes(np.array([3.,2.,1.]),np.array([1.,2.,3.]),np.array([0,0,1]),np.array([0,0,0]))
    assert (changed,total)==(1,1)
