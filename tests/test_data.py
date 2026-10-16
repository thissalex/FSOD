import pytest

from fsod_experiments.data import assert_disjoint, sample_support


def test_instance_shot_and_reproducibility():
    coco = {'images': [{'id': 1, 'file_name': 'image.png'}],
            'categories': [{'id': 7, 'name': 'object'}],
            'annotations': [{'id': i, 'image_id': 1, 'category_id': 7,
                             'bbox': [i, i, 1, 1]} for i in range(10)]}
    a = sample_support(coco, 5, 1)
    assert a == sample_support(coco, 5, 1)
    assert len(a['object']) == 5
    assert len({s['instance_id'] for s in a['object']}) == 5
    with pytest.raises(ValueError, match='Insufficient'):
        sample_support(coco, 11, 1)


def test_cross_path_duplicate_content_is_rejected():
    support = {'images': [{'file_name': 'a.png'}], 'categories': []}
    query = {'images': [{'file_name': 'b.png'}], 'categories': []}
    audit = {'image_content_hashes': ['same-content']}
    with pytest.raises(ValueError, match='content overlap'):
        assert_disjoint(support, query, audit, audit)
