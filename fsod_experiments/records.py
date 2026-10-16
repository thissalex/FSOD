"""Stable content hashes, immutable records and paired support-set bootstrap."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def paired_summary(control, treatment, seed=0, repetitions=10000):
    if set(control) != set(treatment) or not control:
        raise ValueError("Require identical nonempty support-set keys")
    keys = sorted(control)
    delta = np.array([treatment[k] - control[k] for k in keys], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite AP values")
    rng = np.random.default_rng(seed)
    ci = None
    if len(delta) >= 2:
        sampled = rng.choice(delta, (repetitions, len(delta)), replace=True).mean(1)
        ci = np.quantile(sampled, [0.025, 0.975]).tolist()
    return {"support_keys": keys, "paired_delta_ap_points": delta.tolist(),
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else None,
            "ci95_percentile": ci, "positive_fraction": float((delta > 0).mean()),
            "n_support_sets": len(delta)}


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        writer.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows])
