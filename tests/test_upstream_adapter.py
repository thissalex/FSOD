import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from fsod_experiments.upstream_adapter import install


@pytest.mark.parametrize('method', ['B1', 'B2', 'B5', 'Ours'])
def test_adapter_preserves_mask_and_transforms_both_sides(tmp_path, monkeypatch, method):
    config = json.load(open('configs/stage1.json'))
    monkeypatch.chdir(tmp_path)
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (20, 24, 3), dtype=np.uint8)).save('a.png')
    mask_calls = []
    def mask(predictor, image, bbox, device):
        mask_calls.append(bbox)
        result = np.zeros((20, 24), dtype=np.float32)
        result[5:15, 5:15] = 1
        return result
    def raw(model, transform, image, device):
        array = np.array(image).copy().astype(np.float32)/255
        # Add a globally varying component so framework-free test detects context response.
        x = torch.from_numpy(array).permute(2, 0, 1)[None]
        return x + x.mean((2, 3), keepdim=True)
    def prototypes(bank, device):
        return F.normalize(torch.stack([v[0] for v in bank.values()], dim=1), dim=0), list(bank)
    module = SimpleNamespace(get_dinov2_features=raw, extract_masks_from_support=mask,
                             resize_mask_to_features=lambda m, shape: m,
                             extract_support_features=None, compute_prototype_weights=prototypes)
    out = tmp_path/'run';out.mkdir()
    install(module, method, config, 10, 1, out, {'weights': 'test-only'})
    support = {'first': [{'image': 'a.png', 'bbox': [5, 5, 10, 10]}],
               'second': [{'image': 'a.png', 'bbox': [6, 6, 9, 9]}]}
    bank = module.extract_support_features(support, None, 'DINOV2', None, None, '.', 'cpu')
    assert len(mask_calls) == 2  # one SAM call per original instance, never per view
    cached = module.extract_support_features(support, None, 'DINOV2', None, None, '.', 'cpu')
    assert len(mask_calls) == 2  # a cache hit must not regenerate masks or features
    assert all(torch.equal(bank[c][0], cached[c][0]) for c in bank)
    proto, classes = module.compute_prototype_weights(bank, 'cpu')
    assert classes == ['first', 'second']
    if method in ('B5', 'Ours'):
        a = torch.load(out/'transform.pt', weights_only=True).float()
        image = Image.open('a.png')
        before = raw(None, None, image, 'cpu')
        after = module.get_dinov2_features(None, None, image, 'cpu')
        # Applying A to a feature map must commute with fixed region pooling.
        m = torch.from_numpy(mask(None, image, [], 'cpu'))
        pooled_after = (after*m).sum((2, 3))/m.sum()
        pooled_before = (before*m).sum((2, 3))/m.sum()
        assert torch.allclose(pooled_after, pooled_before @ a.T, atol=1e-6)
        assert torch.allclose(torch.linalg.vector_norm(proto, dim=0), torch.ones(2))
    assert (out/'adaptation_diagnostics.json').exists()
