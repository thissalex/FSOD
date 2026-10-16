import json

import numpy as np
import pytest
from PIL import Image

from fsod_experiments.records import digest, paired_summary
from fsod_experiments.views import make_views


def test_pixel_protection_and_nonempty_context():
    config = json.load(open("configs/stage1.json"))
    a = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    views, metadata = make_views(Image.fromarray(a), [10.2, 8.1, 35.7, 30.2], config)
    assert len(views) == 5
    x1, y1, x2, y2 = metadata[0]["protected_xyxy"]
    for image in views:
        assert np.array_equal(np.asarray(image)[y1:y2, x1:x2], a[y1:y2, x1:x2])
    assert not np.array_equal(np.asarray(views[1]), a)
    _, full = make_views(Image.fromarray(a), [0, 0, 60, 40], config)
    assert full[0]["perturbable_fraction"] == 0


def test_paired_bootstrap_and_missing_support():
    stats = paired_summary({"s0": 20, "s1": 30}, {"s0": 21, "s1": 31})
    assert stats["mean_delta"] == 1
    assert stats["ci95_percentile"] == [1, 1]
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        paired_summary({"s0": 20}, {"s1": 21})
