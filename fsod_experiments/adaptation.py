"""Raw-feature prototype aggregation and paired, normalized response statistics."""
import math

import torch
import torch.nn.functional as F


def normalize(x):
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite region features")
    if (torch.linalg.vector_norm(x, dim=-1) < 1e-12).any():
        raise ValueError("Near-zero region feature; inspect empty masks")
    return F.normalize(x, dim=-1, eps=1e-12)


def fit(support, labels, lam=10.0, mu=1.0):
    """support [instances, views, d], labels [instances]; return A, p, classes, diagnostics.

    Prototype aggregation averages RAW features over views then instances, matching
    upstream compute_prototype_weights. Only the statistics branch normalizes views.
    """
    if support.ndim != 3 or min(support.shape) == 0:
        raise ValueError("Expected nonempty [instances, views, dimension]")
    if labels.ndim != 1 or len(labels) != len(support):
        raise ValueError("One category label is required per support instance")
    if not all(math.isfinite(v) and v >= 0 for v in (lam, mu)):
        raise ValueError("lambda and mu must be finite and nonnegative")
    x = support.to(dtype=torch.float64)
    labels = labels.to(x.device)
    z = normalize(x)
    delta = z - z.mean(dim=1, keepdim=True)
    e = delta.flatten(0, 1).T / math.sqrt(x.shape[0] * x.shape[1])
    classes = torch.unique(labels, sorted=True)
    p = normalize(torch.stack([x[labels == c].mean(dim=(0, 1)) for c in classes]))
    pc = (p - p.mean(dim=0)).T / math.sqrt(len(classes))
    c, s = e @ e.T, pc @ pc.T
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    h, b = eye + lam * c + mu * s, eye + mu * s
    a = torch.linalg.solve(h, b).T
    residual = (torch.linalg.norm(a @ h - b) / torch.linalg.norm(b)).item()
    if not torch.isfinite(a).all() or residual > 1e-8:
        raise RuntimeError(f"Invalid solve: relative residual={residual}")
    return a, p, classes, {
        "solve_relative_residual": residual,
        "context_trace": torch.trace(c).item(),
        "class_trace": torch.trace(s).item(),
        "mean_cosine_drift": (1 - (z[:, :1] * z).sum(-1)).mean().item(),
    }


def match(query, prototypes, transform):
    q = normalize(query.to(transform) @ transform.T)
    p = normalize(prototypes.to(transform) @ transform.T)
    return q @ p.T
