import importlib.util
from pathlib import Path

import pytest
from PIL import Image

spec=importlib.util.spec_from_file_location('development',Path(__file__).parents[1]/'scripts/prepare_development.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_partition_keeps_duplicate_group_and_class_coverage():
    data={'images':[{'id':i} for i in range(20)], 'categories':[{'id':0},{'id':1}],
          'annotations':[{'id':i,'image_id':i,'category_id':i%2} for i in range(20)]}
    hashes={str(i):[i] for i in range(2,20)};hashes['duplicate']=[0,1]
    support,query,sc,qc=module.partition(data,hashes,{},max_shot=5)
    assert not support&query
    assert (0 in query)==(1 in query)
    assert min(sc.values())>=5 and all(qc[c]>=1 for c in [0,1])
    assert (support,query)==module.partition(data,hashes,{},max_shot=5)[:2]


def test_impossible_split_fails_instead_of_leaking():
    data={'images':[{'id':i} for i in range(5)], 'categories':[{'id':0}],
          'annotations':[{'id':i,'image_id':i,'category_id':0} for i in range(5)]}
    with pytest.raises(ValueError,match='Cannot isolate'):
        module.partition(data,{str(i):[i] for i in range(5)},{},max_shot=5)


def test_audit_detects_metadata_mismatch_and_decoded_pixel_duplicates(tmp_path):
    image=Image.new('RGB',(10,12),'red')
    image.save(tmp_path/'a.png',compress_level=0)
    image.save(tmp_path/'b.png',compress_level=9)
    data={'images':[{'id':0,'file_name':'a.png','width':10,'height':12},
                    {'id':1,'file_name':'b.png','width':20,'height':24}],
          'annotations':[],'categories':[{'id':0,'name':'object'}]}
    bad,groups=module.audit(data,tmp_path)
    assert bad[1]['reason']=='dimension_mismatch'
    assert len(groups)==1 and sorted(next(iter(groups.values())))==[0,1]
