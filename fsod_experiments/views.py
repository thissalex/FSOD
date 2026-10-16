"""Pixel-exact target protection. All operations use lossless RGB arrays."""
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def protected_box(image, bbox_xyxy, margin=0.1):
    w, h = image.size
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, margin)):
        raise ValueError("Nonfinite protection coordinates")
    if margin < 0 or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Expected valid in-image xyxy box")
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(w, math.ceil(x2 + dx)), min(h, math.ceil(y2 + dy)))


def make_views(image, bbox_xyxy, config):
    image = image.convert("RGB")
    w, h = image.size
    original = np.asarray(image)
    x1, y1, x2, y2 = protected_box(image, bbox_xyxy, config["protection_margin"])
    specs = [("original", 0)]
    specs += [("blur", r) for r in config["blur_radii"]]
    specs += [("brightness", f) for f in config["brightness_factors"]]
    views, metadata = [], []
    for kind, strength in specs:
        changed = image.copy()
        if kind == "blur":
            changed = image.filter(ImageFilter.GaussianBlur(strength))
        elif kind == "brightness":
            changed = ImageEnhance.Brightness(image).enhance(strength)
        array = np.array(changed)
        array[y1:y2, x1:x2] = original[y1:y2, x1:x2]
        if not np.array_equal(array[y1:y2, x1:x2], original[y1:y2, x1:x2]):
            raise AssertionError("Protection invariant violated")
        views.append(Image.fromarray(array))
        metadata.append({"type": kind, "strength": strength,
                         "protected_xyxy": [x1, y1, x2, y2],
                         "perturbable_fraction": 1 - (x2-x1)*(y2-y1)/(w*h)})
    if len(views) != config["extra_views"] + 1:
        raise ValueError("Configured view count does not match perturbations")
    return views, metadata
