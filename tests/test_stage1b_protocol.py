import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_nested_budget_clips_after_truncation():
    proposal=module('proposal_budget_stage1b')
    boxes=[[0,0,5,5],[9,0,15,5],[12,0,15,5],[-1,-1,3,3]]
    small=proposal.clipped_prefix(boxes,2,10,10)
    large=proposal.clipped_prefix(boxes,4,10,10)
    assert np.array_equal(large[:len(small)],small)
    assert len(large)==3  # invalid third box removed, fourth retained at larger budget
    assert np.array_equal(small[1],[9,0,1,5])


def test_phash_preserves_resize_and_distinguishes_content():
    audit=module('audit_stage1b');rng=np.random.default_rng(4)
    a=Image.fromarray(rng.integers(0,256,(128,128),dtype=np.uint8))
    b=a.resize((256,256),Image.Resampling.NEAREST)
    assert (audit.phash(a)!=audit.phash(b)).sum()<=2
    c=Image.fromarray(rng.integers(0,256,(128,128),dtype=np.uint8))
    assert (audit.phash(a)!=audit.phash(c)).sum()>16
