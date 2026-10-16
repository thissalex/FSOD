#!/usr/bin/env python3
"""Additional translation/brightness audit after pHash missed a shifted duplicate.

NCC >= .95 on centered 48x48 crops of 64x64 grayscale images nominates
candidates only; visual + original-resolution alignment determine exclusion.
No detector outputs or labels enter candidate selection.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import write_json

def main():
    cv2.setNumThreads(1);reports=[]
    for domain in ['NEU-DET','clipart1k']:
        base=ROOT/'artifacts/development'/domain;root=ROOT/'dataset/CDFSOD'/domain/'train'
        pool=json.loads((base/'support_pool.json').read_text())['images'];query=json.loads((base/'query.json').read_text())['images']
        def read(im):return cv2.resize(cv2.imread(str(root/im['file_name']),0),(64,64),interpolation=cv2.INTER_AREA)
        support=[read(im) for im in pool];pairs=[]
        for j,im in enumerate(query):
            template=read(im)[8:56,8:56]
            if template.std()<1:continue
            for i,a in enumerate(support):
                _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(a,template,cv2.TM_CCOEFF_NORMED))
                if score>=.95:pairs.append({'support_image':pool[i],'query_image':im,'ncc64':score,'offset64':loc})
        pairs.sort(key=lambda p:-p['ncc64'])
        artifact=ROOT/'artifacts/stage1b/audit'
        for page in range((len(pairs)+9)//10):
            sheet=Image.new('RGB',(760,1700),'white');draw=ImageDraw.Draw(sheet)
            for row,pair in enumerate(pairs[page*10:page*10+10]):
                for col,key in enumerate(['support_image','query_image']):
                    with Image.open(root/pair[key]['file_name']) as image:thumb=ImageOps.contain(image.convert('RGB'),(350,140))
                    sheet.paste(thumb,(col*380,row*170+25));draw.text((col*380,row*170+3),f"#{page*10+row} {pair['ncc64']:.3f} {pair[key]['file_name']}",fill='black')
            sheet.save(artifact/f'{domain}_registration_{page}.jpg')
        reports.append({'domain':domain,'threshold_ncc64':.95,'pairs':pairs});print(domain,len(pairs),flush=True)
    write_json(ROOT/'results/stage1b/registration_audit.json',reports)
if __name__=='__main__':main()
