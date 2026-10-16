"""Enforce exact target pixels after both official preprocessing resizes.

Grow only the protected source-image region by a deterministic pixel guard when
interpolation leaks across its boundary. No labels, AP or feature values choose
the guard. Original perturbation strengths, image geometry and masks stay fixed.
"""
import numpy as np


def protected_views(image,bbox_xyxy,config,make_views,resize,transform):
    bx,by,x_end,y_end=bbox_xyxy;iw,ih=image.size
    original_input=transform(resize(image,630,14))
    th,tw=original_input.shape[-2:]
    x1,y1=int(np.ceil(bx/iw*tw)),int(np.ceil(by/ih*th))
    x2,y2=int(np.floor(x_end/iw*tw)),int(np.floor(y_end/ih*th))
    if x2<=x1 or y2<=y1:
        raise ValueError('Target has no full input pixel after preprocessing')
    for padding in [0,1,2,4,8,16,max(iw,ih)]:
        protected=[max(0,bx-padding),max(0,by-padding),min(iw,x_end+padding),min(ih,y_end+padding)]
        views,metadata=make_views(image,protected,config)
        leaks=[]
        for view in views:
            value=transform(resize(view,630,14))
            leaks.append(float((value-original_input)[:,y1:y2,x1:x2].abs().max()))
        if not any(leaks):
            return views,metadata,leaks,padding
    raise RuntimeError('Could not enforce target pixel protection')
