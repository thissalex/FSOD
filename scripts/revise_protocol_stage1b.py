#!/usr/bin/env python3
"""Apply reviewed content-based query exclusions; never modify v1 or test files."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import write_json,file_digest


def main():
    reviewed=json.loads((ROOT/'results/stage1b/duplicate_review.json').read_text());revisions=[]
    for domain in ['NEU-DET','clipart1k']:
        excluded={r['query_file'] for r in reviewed if r['domain']==domain and r['decision']=='exclude_query'}
        old=ROOT/'artifacts/development'/domain/'query.json';data=json.loads(old.read_text());ids={i['id'] for i in data['images'] if i['file_name'] in excluded}
        if len(ids)!=len(excluded):raise ValueError('Unknown or duplicated excluded filename')
        data['images']=[i for i in data['images'] if i['id'] not in ids];data['annotations']=[a for a in data['annotations'] if a['image_id'] not in ids]
        output=ROOT/'artifacts/development_v2'/domain/'query.json';write_json(output,data)
        original=json.loads(old.read_text())
        revisions.append({'domain':domain,'protocol':'v2','old_query_sha256':file_digest(old),'query_sha256':file_digest(output),
            'query_relative_path':str(output.relative_to(ROOT)),'excluded_query_files':sorted(excluded),'excluded_query_image_ids':sorted(ids),
            'old_query_images':len(original['images']),'query_images':len(data['images']),'old_query_instances':len(original['annotations']),
            'query_instances':len(data['annotations']),'support_pool_changed':False,'support_manifests_changed':False,
            'rule':'Exclude the query side of confirmed cross-partition same-content groups; do not move into support or resample. No AP-based exclusions.'})
    write_json(ROOT/'results/stage1b/protocol_revision.json',revisions)
if __name__=='__main__':main()
