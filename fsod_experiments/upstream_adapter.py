"""Development adapter for an unmodified, pinned FSOD-VFM main.py.

The shared linear transform commutes with fixed mask pooling. Applying it to
query dense features therefore reuses the official matching and graph code.
Region-level query caching and official-vs-adapter parity on real data remain
separate acceptance checks; this initial runner does not claim those are done.
"""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .adaptation import fit, normalize
from .records import digest, file_digest, write_json
from .views import make_views


def install(support_util, method, config, lam, mu, output_dir, cache_identity):
    original_extract = support_util.extract_support_features
    original_prototypes = support_util.compute_prototype_weights
    raw_dino = support_util.get_dinov2_features
    state = {'method': method, 'query_cache': 'not_implemented', 'empty_masks': []}
    output_dir = Path(output_dir)

    def extract(data, predictor, name, model, transform, data_dir, device='cpu'):
        if name != 'DINOV2':
            raise ValueError('Development adapter currently supports DINOv2 only')
        if method == 'B0':
            return original_extract(data, predictor, name, model, transform, data_dir, device)
        started = time.perf_counter()
        features, labels, instances = [], [], []
        names = sorted(data)
        for label, cls in enumerate(names):
            for sample in data[cls]:
                path = Path(data_dir) / sample['image']
                image = Image.open(path).convert('RGB')
                x, y, w, h = sample['bbox']
                identity = {**cache_identity, 'sample': sample, 'image_sha256': file_digest(path),
                            'view_config': config, 'mask': 'original_sam2_fixed',
                            'pooling': 'official_cv2_threshold_0.5_raw_fp32'}
                key = digest(identity)
                cache = Path('cache/support') / (key + '.npz')
                cache.parent.mkdir(parents=True, exist_ok=True)
                if cache.exists():
                    with np.load(cache, allow_pickle=False) as loaded:
                        vectors = torch.from_numpy(loaded['features'].copy())
                        repeat = loaded['repeat'].copy()
                        meta = json.loads(str(loaded['metadata']))
                else:
                    views, meta = make_views(image, [x, y, x+w, y+h], config)
                    mask = support_util.extract_masks_from_support(predictor, image, [x, y, x+w, y+h], device)
                    vectors = []
                    fixed_mask = None
                    for view in views + [image.copy()]:
                        feat = raw_dino(model, transform, view, device)
                        if fixed_mask is None:
                            projected = support_util.resize_mask_to_features(mask, feat.shape[2:])
                            fixed_mask = torch.from_numpy(projected).to(device)[None, None]
                            if fixed_mask.sum() == 0:
                                raise RuntimeError(f'Empty support mask: {sample}; cannot compare changed shot counts')
                        pooled = (feat * fixed_mask).sum((2, 3)) / fixed_mask.sum()
                        vectors.append(pooled[0].detach().float().cpu())
                    repeat = vectors.pop().numpy()
                    vectors = torch.stack(vectors)
                    temp = cache.with_suffix('.tmp.npz')
                    np.savez_compressed(temp, features=vectors.numpy(), repeat=repeat,
                                        metadata=json.dumps(meta))
                    temp.replace(cache)
                    write_json(cache.with_suffix('.json'), identity)
                normalize(vectors)
                z = normalize(vectors.double())
                noise = 1 - (z[0] * normalize(torch.from_numpy(repeat).double())).sum().item()
                instances.append({'category': cls, 'sample': sample, 'cache_key': key,
                                  'views': meta, 'repeat_cosine_noise': noise,
                                  'view_cosine_drift': (1 - (z[0] * z).sum(-1)).tolist()})
                # B2 is one fixed strong outside-box blur; all candidates are cached
                # once for equal auditability, but B2 has no transformation.
                if method == 'B2':
                    index = next(i for i, v in enumerate(meta)
                                 if v['type'] == 'blur' and v['strength'] == max(config['blur_radii']))
                    vectors = vectors[index:index+1]
                features.append(vectors)
                labels.append(label)
        stacked = torch.stack(features)
        a, p, classes, stats = fit(stacked, torch.tensor(labels),
                                 lam if method in ('B5', 'Ours') else 0,
                                 mu if method == 'Ours' else 0)
        state.update(stats)
        state['support_elapsed_seconds'] = time.perf_counter() - started
        state['support_instances'] = instances
        state['transform'] = a
        # Upstream compute_prototype_weights expects raw prototype per class.
        return {names[int(c)]: [p[i].float()] for i, c in enumerate(classes)}

    def prototypes(memory_bank, device):
        p, names = original_prototypes(memory_bank, device)
        if method in ('B5', 'Ours'):
            a = state['transform'].to(device=device, dtype=p.dtype)
            p = F.normalize(a @ p, dim=0, eps=1e-12)
            def query_features(model, transform, image, device='cpu'):
                raw = raw_dino(model, transform, image, device)
                return torch.einsum('oc,bchw->bohw', a.to(raw), raw)
            # Installed only AFTER support extraction, so support stats stay raw.
            support_util.get_dinov2_features = query_features
        serializable = {k: v for k, v in state.items() if k != 'transform'}
        write_json(output_dir / 'adaptation_diagnostics.json', serializable)
        if 'transform' in state:
            torch.save(state['transform'], output_dir / 'transform.pt')
        return p, names

    support_util.extract_support_features = extract
    support_util.compute_prototype_weights = prototypes
