#!/usr/bin/env python3
"""Record explicit experiment provenance; never inspect environment secrets."""
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fsod_experiments.records import file_digest,write_json


def main():
    sources={}
    for component in ['FSOD-VFM','sam2']:
        path=ROOT/'external'/component;manifest=json.loads((path/'source_manifest.json').read_text())
        for item in manifest['files']:
            body=(path/item['path']).read_bytes()
            if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=item['sha']:raise ValueError(f'Modified source {item["path"]}')
        sources[component]={'commit':manifest['commit'],'verified_git_blobs':len(manifest['files']),
                            'manifest_sha256':file_digest(path/'source_manifest.json'),'local_source_changes':[]}
    dino=ROOT/'external/dinov2'
    sources['dinov2']={'commit':subprocess.check_output(['git','-C',str(dino),'rev-parse','HEAD'],text=True).strip(),
        'tracked_changes':subprocess.check_output(['git','-C',str(dino),'status','--porcelain','--untracked-files=no'],text=True).strip()}
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z'],text=True).split('\0')
    files=[ROOT/p for p in tracked if p and Path(p).suffix in ['.py','.json'] and Path(p).parts[0] in ['fsod_experiments','scripts','configs','tests']]
    manifest={'requirements_sha256':file_digest(ROOT/'requirements.txt'),'plan_sha256':file_digest(ROOT/'少样本目标检测_阶段一评估与下一步工作计划.md'),'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True).strip(),
        'native_extension_sha256':{str(p.relative_to(ROOT)):file_digest(p) for folder in ['external/FSOD-VFM/chatrex/upn/ops','external/sam2/sam2'] for p in (ROOT/folder).glob('*.so')},'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'inference_precision':'FP32, no autocast','xformers_installed':any(d.metadata['Name'].lower()=='xformers' for d in distributions()),
        'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in distributions()],key=lambda d:d['name'].lower()),
        'sources':sources,'source_sha256':{str(p.relative_to(ROOT)):file_digest(p) for p in files},
        'weight_sha256':{p.name:file_digest(p) for p in sorted((ROOT/'checkpoints').glob('*')) if p.suffix in ['.pt','.pth']},
        'archives_recorded_manifest':'results/resume/archive_manifest.json',
        'archive_sizes_current':{p.name:p.stat().st_size for p in sorted((ROOT/'CDFSOD-benchmark').glob('*.tar.gz'))},
        'archive_digest_note':'SHA256 previously verified in stage1; size checked now; no re-download or new extraction of confirmation domains',
        'protocol_v1':'results/resume/development_protocol.json','protocol_v2':'results/stage1b/protocol_revision.json',
        'graph':{'steps':30,'alpha':.3,'score_decay_exponent':.5,'max_proposals':100,'coco_maxDets':[1,10,100]},
        'guard':'original 10% protect band, fixed original SAM mask, extra deterministic interpolation padding selected by exact pixels only'}
    manifests=[]
    for domain in ['NEU-DET','clipart1k']:
        unique=set();appearances=0
        for seed in [0,1,2]:
            path=ROOT/'artifacts/development'/domain/f'support_5shot_seed{seed}.json';samples=json.loads(path.read_text())
            for cls,items in samples.items():
                for i in items:unique.add((i['image'],i['instance_id']));appearances+=1
            manifests.append({'domain':domain,'seed':seed,'shot':5,'sha256':file_digest(path),'categories':sorted(samples)})
        manifests.append({'domain':domain,'scope':'5shot seeds0-2','instance_appearances':appearances,'unique_instances':len(unique),'views_per_appearance':5})
    manifest['support_index']=manifests
    manifest['heldout_audit']={'seeds':list(range(3,10)), 'prior_method_result_seeds':sorted({r['seed'] for domain in ['NEU-DET','clipart1k'] for filename in ['paired_results.json','prespecified_grid.json'] for r in json.loads((ROOT/'results/resume'/domain/filename).read_text())}),
        'scope':'Checked recorded core/grid result files; no heldout method outputs found. Manifests only were generated; no claim about unrecorded external activity.'}
    write_json(ROOT/'results/stage1b/repro_manifest.json',manifest)
if __name__=='__main__':main()
