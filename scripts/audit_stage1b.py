#!/usr/bin/env python3
"""Cross-partition perceptual candidates; decisions require human/visual review."""
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fsod_experiments.records import write_json,file_digest

def phash(image):
    values=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),float)
    low=dctn(values,norm='ortho')[:8,:8].flatten()
    return low > np.median(low[1:])

def main():
    config=json.loads((ROOT/'configs/stage1b.json').read_text()); reports=[]
    artifact=ROOT/'artifacts/stage1b/audit';artifact.mkdir(parents=True,exist_ok=True)
    for domain in config['domains']:
        base=ROOT/'artifacts/development'/domain
        pool=json.loads((base/'support_pool.json').read_text())
        query=json.loads((base/'query.json').read_text())
        root=ROOT/'dataset/CDFSOD'/domain/'train'
        images=pool['images']+query['images']; hashes=[]
        for im in images:
            with Image.open(root/im['file_name']) as image:hashes.append(phash(image))
        hashes=np.stack(hashes); n=len(pool['images']); pairs=[];nearest=[]
        for j,im in enumerate(query['images']):
            distances=(hashes[:n]!=hashes[n+j]).sum(1)
            nearest.append({'support_image':pool['images'][int(distances.argmin())], 'query_image':im,'hamming':int(distances.min())})
            for i in np.flatnonzero(distances<=config['phash_hamming_max']):
                pairs.append({'support_image':pool['images'][i], 'query_image':im,'hamming':int(distances[i])})
        pairs.sort(key=lambda x:x['hamming'])
        nearest.sort(key=lambda x:x['hamming']);review=pairs if pairs else nearest[:10]
        for page in range((len(review)+9)//10):
            sheet=Image.new('RGB',(760,10*170),'white');draw=ImageDraw.Draw(sheet)
            for row,pair in enumerate(review[page*10:page*10+10]):
                for col,key in enumerate(['support_image','query_image']):
                    with Image.open(root/pair[key]['file_name']) as image:
                        thumb=ImageOps.contain(image.convert('RGB'),(350,140))
                    sheet.paste(thumb,(col*380,row*170+25))
                    draw.text((col*380,row*170+3),f"#{page*10+row} d={pair['hamming']} {pair[key]['file_name']}",fill='black')
            sheet.save(artifact/f'{domain}_{page}.jpg')
        keys=Counter(k for im in images for k in im)
        reports.append({'domain':domain,'support_images':n,'query_images':len(query['images']),
                        'support_sha256':file_digest(base/'support_pool.json'),'query_sha256':file_digest(base/'query.json'),
                        'image_metadata_keys':dict(keys),'cross_partition_phash_candidates':pairs,
                        'nearest_pairs_for_visual_sanity_check':nearest[:10],'threshold':config['phash_hamming_max'],'source_sequences':'unknown; no verified acquisition/sequence mapping'})
        print(domain,'candidates',len(pairs),flush=True)
    write_json(ROOT/'results/stage1b/perceptual_audit.json',reports)
if __name__=='__main__':main()
