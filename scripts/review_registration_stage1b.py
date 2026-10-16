#!/usr/bin/env python3
"""Original-resolution texture check of translated NCC candidates (NEU only).

Keep original crops at >=.95 high-pass NCC or >=.99 raw NCC for visual review.
These nominations are separate from the human/agent visual decisions.
"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import write_json

def main():
    reports=json.loads((ROOT/'results/stage1b/registration_audit.json').read_text());selected=[]
    for report in reports:
        if report['domain']!='NEU-DET':continue
        root=ROOT/'dataset/CDFSOD/NEU-DET/train'
        for i,pair in enumerate(report['pairs']):
            a=cv2.imread(str(root/pair['support_image']['file_name']),0).astype(np.float32);b=cv2.imread(str(root/pair['query_image']['file_name']),0).astype(np.float32)
            ha=a-cv2.GaussianBlur(a,(0,0),3);hb=b-cv2.GaussianBlur(b,(0,0),3)
            _,raw,_,loc=cv2.minMaxLoc(cv2.matchTemplate(a,b[25:-25,25:-25],cv2.TM_CCOEFF_NORMED))
            _,high,_,hloc=cv2.minMaxLoc(cv2.matchTemplate(ha,hb[25:-25,25:-25],cv2.TM_CCOEFF_NORMED))
            pair.update(original_raw_ncc=raw,original_highpass_ncc=high,original_match_location=loc,original_highpass_match_location=hloc)
            if raw>=.99 or high>=.95:
                selected.append(pair);print(pair['support_image']['file_name'],pair['query_image']['file_name'],raw,high,loc,flush=True)
    for page in range((len(selected)+9)//10):
        sheet=Image.new('RGB',(760,1700),'white');draw=ImageDraw.Draw(sheet)
        for row,pair in enumerate(selected[page*10:page*10+10]):
            for col,key in enumerate(['support_image','query_image']):
                with Image.open(ROOT/'dataset/CDFSOD/NEU-DET/train'/pair[key]['file_name']) as image:thumb=ImageOps.contain(image.convert('RGB'),(350,140))
                sheet.paste(thumb,(col*380,row*170+25));draw.text((col*380,row*170+3),f"#{page*10+row} {pair[key]['file_name']}",fill='black')
        sheet.save(ROOT/'artifacts/stage1b/audit'/f'confirmed_review_{page}.jpg')
    write_json(ROOT/'results/stage1b/registration_audit.json',reports)
    write_json(ROOT/'results/stage1b/texture_review_candidates.json',selected)
if __name__=='__main__':main()
