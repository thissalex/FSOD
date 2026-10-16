import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location('cached_detection',Path(__file__).parents[1]/'fsod_experiments/cached_detection.py')
cached=importlib.util.module_from_spec(spec);spec.loader.exec_module(cached)


def fake_original(images):
    result=[]
    for img_dict in images:
        img_id=img_dict['id']
        img_results=[{'x':img_id}]
        if img_results:
            img_results[0]['x']+=10
        if img_results:
            result.extend(img_results)
    return result


def test_instrument_captures_before_graph_without_changing_output():
    captures=[]
    adapted=cached.instrument(fake_original,lambda i,items:captures.append((i,items[0]['x'])))
    assert adapted([{'id':7}])==fake_original([{'id':7}])
    assert captures==[(7,7)]


def test_replay_regroups_after_category_change_and_keeps_order(tmp_path):
    features=np.array([[1,0],[0,1],[0.8,0.6]],dtype=np.float32)
    masks=np.ones((3,1,1,2,2),dtype=np.uint8)
    boxes=np.array([[5,0,2,2],[1,0,2,2],[9,0,2,2]])
    np.savez(tmp_path/'1.npz',features=features,masks=masks,boxes=boxes)
    groups=[]
    def graph(m,labels,n,**kwargs):
        groups.append(labels.tolist());return torch.zeros(len(labels))
    p=torch.eye(2)
    before=cached.replay(tmp_path,[1],p,[10,20],graph,device='cpu')
    cached.replay(tmp_path,[1],p.flip(0),[10,20],graph,device='cpu')
    assert groups[0]!=groups[1]
    assert {tuple(i['bbox']) for i in before}=={tuple(i) for i in boxes}
    assert cached.assert_parity(before,before)['passed']
    bad=[{**before[0],'score':before[0]['score']+0.1}]+before[1:]
    with pytest.raises(AssertionError,match='score error'):
        cached.assert_parity(before,bad)
