#!/usr/bin/env python3
"""Audit local benchmark archives and isolate development data from training only.

Preserve original annotations. Exclude inconsistent training images from the
development pool, group identical decoded RGB images, and preserve >=K support
instances and >=1 query instance per category. Test annotations are audited but
never used for partition selection. Manifests are local artifacts, not data uploads.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def audit(data, root):
    cats = {c['id']:c['name'] for c in data['categories']}
    images = {i['id']:i for i in data['images']}
    if len(images) != len(data['images']) or len(cats) != len(data['categories']):
        raise ValueError('Duplicate image/category IDs')
    if len({a['id'] for a in data['annotations']}) != len(data['annotations']):
        raise ValueError('Duplicate annotation IDs')
    invalid, hashes = {}, defaultdict(list)
    for im in images.values():
        path = (root/im['file_name']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Image escapes root')
        with Image.open(path) as original:
            rgb = original.convert('RGB')
            h = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            hashes[h].append(im['id'])
            if rgb.size != (im['width'],im['height']):
                invalid[im['id']] = {'file':im['file_name'], 'reason':'dimension_mismatch',
                                      'declared':[im['width'],im['height']], 'actual':list(rgb.size)}
    for ann in data['annotations']:
        if ann['image_id'] not in images or ann['category_id'] not in cats:
            raise ValueError('Unmapped annotation image/category')
        x,y,w,h = ann['bbox']; im=images[ann['image_id']]
        if not (0<=x<x+w<=im['width'] and 0<=y<y+h<=im['height']):
            invalid[im['id']]={'file':im['file_name'],'reason':'invalid_bbox','annotation_id':ann['id']}
    return invalid, dict(hashes)


def partition(data, hashes, excluded, seed=20261016, max_shot=5, fraction=0.2):
    """Greedily reserve class coverage, then grow query without exhausting support."""
    by_image=defaultdict(Counter)
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['image_id'] not in excluded:
            by_image[a['image_id']][a['category_id']]+=1
    groups=[set(ids) for _,ids in sorted(hashes.items()) if not set(ids)&set(excluded)]
    random.Random(seed).shuffle(groups)
    cats=sorted(c['id'] for c in data['categories'])
    remaining=sum((by_image[i] for group in groups for i in group), Counter())
    qcounts=Counter(); query=set()
    def take(group):
        counts=sum((by_image[i] for i in group),Counter())
        if any(remaining[c]-counts[c]<max_shot for c in cats):return False
        query.update(group);qcounts.update(counts);remaining.subtract(counts)
        return True
    for cat in sorted(cats,key=lambda c:remaining[c]):
        if qcounts[cat]:continue
        if not any(take(g) for g in groups if not (g&query) and any(by_image[i][cat] for i in g)):
            raise ValueError(f'Cannot isolate class {cat} with {max_shot} support and query coverage')
    target=round((len(data['images'])-len(excluded))*fraction)
    for group in groups:
        if len(query)>=target:break
        if not group&query:take(group)
    support={i['id'] for i in data['images']}-set(excluded)-query
    if support&query:raise AssertionError('Image leakage')
    for ids in hashes.values():
        if set(ids)&query and set(ids)&support:raise AssertionError('Duplicate-content leakage')
    return support,query,remaining,qcounts


def subset(data, ids):
    return {**data,'images':[i for i in data['images'] if i['id'] in ids],
            'annotations':[a for a in data['annotations'] if a['image_id'] in ids]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='dataset/CDFSOD')
    p.add_argument('--output',default='artifacts/development');p.add_argument('--seed',type=int,default=20261016)
    args=p.parse_args(); reports=[]
    for domain in ['NEU-DET','clipart1k']:
        root=Path(args.root)/domain
        train=json.loads((root/'annotations/train.json').read_text())
        test=json.loads((root/'annotations/test.json').read_text())
        bad,hashes=audit(train,root/'train');test_bad,test_hashes=audit(test,root/'test')
        # Do not silently alter official test data when auditing identifies a problem.
        if test_bad:raise ValueError(f'Official test annotation inconsistency: {test_bad}')
        overlap=set(hashes)&set(test_hashes)
        if overlap:raise ValueError('Decoded-pixel train/test leakage')
        support,query,sc,qc=partition(train,hashes,bad,args.seed)
        out=Path(args.output)/domain
        write(out/'support_pool.json',subset(train,support));write(out/'query.json',subset(train,query))
        images={i['id']:i for i in train['images']}
        anns={a['id']:a for a in train['annotations']}
        support_manifests=[]
        for shot in [1,5]:
            for seed in range(10):
                rng=random.Random(seed);selected={}
                for cat in sorted(train['categories'],key=lambda c:c['id']):
                    pool=sorted([a for a in anns.values() if a['image_id'] in support and
                                 a['category_id']==cat['id'] and not a.get('iscrowd',0)],key=lambda a:a['id'])
                    selected[cat['name']]=[{'image':f'CDFSOD/{domain}/train/'+images[a['image_id']]['file_name'],
                                           'bbox':a['bbox'],'instance_id':a['id'],'image_id':a['image_id'],
                                           'category_id':cat['id']}
                                          for a in rng.sample(pool,shot)]
                dest=out/f'support_{shot}shot_seed{seed}.json';write(dest,selected)
                support_manifests.append({'shot':shot,'seed':seed,'sha256':sha(dest)})
        report={'domain':domain,'partition_seed':args.seed,'max_support_shot':5,
                'source':'official train.json only; original official test untouched',
                'train_images':len(train['images']),'test_images':len(test['images']),
                'excluded_training_images':list(bad.values()),
                'train_duplicate_groups':[[images[i]['file_name'] for i in ids] for ids in hashes.values() if len(ids)>1],
                'test_duplicate_group_count':sum(len(ids)>1 for ids in test_hashes.values()),
                'cross_split_pixel_duplicates':len(overlap),'support_pool_images':len(support),
                'development_query_images':len(query),'support_instances_by_category':dict(sc),
                'query_instances_by_category':dict(qc),'support_pool_sha256':sha(out/'support_pool.json'),
                'query_sha256':sha(out/'query.json'),'support_manifests':support_manifests,
                'near_duplicate_review':'Only exact decoded-pixel duplicate groups checked; perceptual/sequence review pending'}
        reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
    write('results/resume/development_protocol.json',reports)


if __name__=='__main__':main()
