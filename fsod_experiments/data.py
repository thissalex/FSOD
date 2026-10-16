"""COCO auditing and support-only sampling; no test-label-driven split creation."""
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from .records import digest, file_digest


def audit_coco(coco, image_root):
    images = {im['id']: im for im in coco['images']}
    cats = {c['id']: c['name'] for c in coco['categories']}
    if len(images) != len(coco['images']) or len(cats) != len(coco['categories']):
        raise ValueError('Duplicate image/category IDs')
    if len(set(cats.values())) != len(cats):
        raise ValueError('Duplicate category names')
    if len({a['id'] for a in coco['annotations']}) != len(coco['annotations']):
        raise ValueError('Duplicate annotation IDs')
    root = Path(image_root).resolve()
    content = {}
    for im in images.values():
        path = (root / im['file_name']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image outside root')
        with Image.open(path) as img:
            if img.size != (im['width'], im['height']):
                raise ValueError(f'Image dimension mismatch: {path}')
        content[im['id']] = file_digest(path)
    counts = Counter()
    for ann in coco['annotations']:
        if ann['image_id'] not in images or ann['category_id'] not in cats:
            raise ValueError('Unmapped image/category ID')
        x, y, w, h = ann['bbox']
        im = images[ann['image_id']]
        if not (w > 0 and h > 0 and 0 <= x < x+w <= im['width'] and
                0 <= y < y+h <= im['height']):
            raise ValueError(f'Invalid COCO xywh box: {ann["id"]}')
        if not ann.get('iscrowd', 0):
            counts[cats[ann['category_id']]] += 1
    return {'images': len(images), 'instances_by_class': dict(counts),
            'image_content_hashes': sorted(content.values()), 'coco_hash': digest(coco)}


def sample_support(coco, shot, seed):
    """Sample exact K instances per class from an already isolated SUPPORT POOL."""
    if shot < 1:
        raise ValueError('shot must be positive')
    images = {im['id']: im for im in coco['images']}
    rng = random.Random(seed)
    samples = {}
    for cat in sorted(coco['categories'], key=lambda c: c['id']):
        pool = sorted([a for a in coco['annotations']
                       if a['category_id'] == cat['id'] and not a.get('iscrowd', 0)],
                      key=lambda a: a['id'])
        if len(pool) < shot:
            raise ValueError(f'Insufficient support instances for {cat["name"]}')
        selected = rng.sample(pool, shot)
        samples[cat['name']] = [
            {'image': images[a['image_id']]['file_name'], 'bbox': a['bbox'],
             'instance_id': a['id'], 'image_id': a['image_id'], 'category_id': cat['id']}
            for a in selected]
    return samples


def assert_disjoint(support_coco, query_coco, support_audit, query_audit):
    support_paths = {i['file_name'] for i in support_coco['images']}
    query_paths = {i['file_name'] for i in query_coco['images']}
    if support_paths & query_paths:
        raise ValueError('Support/query file-name overlap')
    if set(support_audit['image_content_hashes']) & set(query_audit['image_content_hashes']):
        raise ValueError('Support/query image-content overlap')
    support_groups = {i['group_id'] for i in support_coco['images'] if 'group_id' in i}
    query_groups = {i['group_id'] for i in query_coco['images'] if 'group_id' in i}
    if support_groups & query_groups:
        raise ValueError('Support/query sequence or duplicate-group overlap')
    if {c['id']: c['name'] for c in support_coco['categories']} != {
            c['id']: c['name'] for c in query_coco['categories']}:
        raise ValueError('Support/query category mapping mismatch')
