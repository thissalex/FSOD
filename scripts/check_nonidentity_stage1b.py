#!/usr/bin/env python3
"""Independent live transformed-dense inference vs fixed-candidate cache replay."""
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/FSOD-VFM'),str(ROOT/'scripts')]
from fsod_experiments.adaptation import fit
from fsod_experiments.cached_detection import replay,assert_parity
from fsod_experiments.records import write_json,file_digest
from diagnose_stage1b import load_batch


def main():
    torch.set_num_threads(4);torch.manual_seed(0);started=time.perf_counter()
    config=json.loads((ROOT/'configs/stage1b.json').read_text());setting=config['nonidentity_check']
    os.chdir(ROOT/'external/FSOD-VFM')
    from model.dinov2 import load_dinov2_model
    from model.sam2 import load_sam2_components
    from chatrex.upn import UPNWrapper
    import support_util as util
    import metric
    model,transform=load_dinov2_model('cuda','dinov2_vitl14',(630,630),str(ROOT/'external/dinov2'),False)
    model.load_state_dict(torch.load(ROOT/'checkpoints/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    _,sam,_=load_sam2_components('large','cuda');upn=UPNWrapper(str(ROOT/'checkpoints/upn_large.pth'))
    reports=[];original_extractor=util.get_dinov2_features
    for domain in config['domains']:
        query=json.loads((ROOT/'artifacts/development_v2'/domain/'query.json').read_text());query['info']=query.get('info',{})
        images=query['images'][:setting['images_per_domain']];ids=[i['id'] for i in images]
        query['images']=images;query['annotations']=[a for a in query['annotations'] if a['image_id'] in ids]
        gt=COCO();gt.dataset=query;gt.createIndex()
        x,y,names,_=load_batch(domain,0);a,p,_,stats=fit(x,y,setting['lambda'],setting['mu'])
        af=a.float().cuda();p=p.float().cuda();transformed_p=F.normalize(p@af.T,dim=1,eps=1e-12)
        def transformed_dense(*args,**kwargs):
            dense=original_extractor(*args,**kwargs)
            return torch.einsum('ij,bjhw->bihw',af,dense)
        util.get_dinov2_features=transformed_dense
        try:
            with torch.inference_mode():
                live=metric.generate_coco_style_predictions_upn(gt,str(ROOT/'dataset/CDFSOD'/domain/'train'),sam,'DINOV2',model,transform,
                     transformed_p.T,names,upn,30,.3,.5,'cuda',.01)
        finally:util.get_dinov2_features=original_extractor
        validation=json.loads((ROOT/'results/resume'/domain/'query_cache_validation.json').read_text())
        catids={c['name']:c['id'] for c in query['categories']}
        cached=replay(ROOT/'cache/stage1_query'/validation['key'],ids,p,[catids[n] for n in names],metric.graph_diffusion_ios,a)
        report={'domain':domain,'seed':0,'shot':5,'lambda':setting['lambda'],'mu':setting['mu'],'image_ids':ids,
                'query_cache_key':validation['key'],'path':'live UPN/SAM/DINO; A applied to dense channels before fixed-mask pooling; transformed prototypes; official class regrouping and diffusion',
                **assert_parity(live,cached,config['parity_tolerance']),**stats}
        reports.append(report);print(json.dumps(report),flush=True)
        write_json(ROOT/'results/stage1b/nonidentity_parity.json',reports)
    torch.cuda.synchronize()
    write_json(ROOT/'results/stage1b/nonidentity_status.json',{'status':'completed','seconds':time.perf_counter()-started,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'source_sha256':file_digest(Path(__file__))})
if __name__=='__main__':main()
