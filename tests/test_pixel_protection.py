import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from pixel_protection import protected_views
from fsod_experiments.views import make_views


def test_guard_repairs_preprocessing_leak_without_changing_original_target():
    image=Image.fromarray(np.random.default_rng(3).integers(0,256,(30,30,3),dtype=np.uint8))
    config={'protection_margin':0.1,'blur_radii':[2,5],'brightness_factors':[0.6,1.4],'extra_views':4}
    resize=lambda im,*args:im.resize((14,14),Image.Resampling.BICUBIC)
    transform=transforms.Compose([transforms.Resize((42,42)),transforms.ToTensor()])
    views,metadata,leaks,padding=protected_views(image,[10,10,13,13],config,make_views,resize,transform)
    assert padding>0 and max(leaks)==0
    for view in views:
        assert np.array_equal(np.asarray(view)[10:13,10:13],np.asarray(image)[10:13,10:13])
    assert metadata[0]['perturbable_fraction']>0
