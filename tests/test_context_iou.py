import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('context_errors',Path(__file__).parents[1]/'scripts/analyze_context_errors.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_candidate_iou_handles_disjoint_and_partial_boxes():
    value=module.iou_xywh([[0,0,10,10]],[[0,0,10,10],[5,0,10,10],[20,0,1,1]])
    assert np.allclose(value,[[1,1/3,0]])
