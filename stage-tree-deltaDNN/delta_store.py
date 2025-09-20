# delta_store.py
import numpy as np
from typing import Dict

def load_npz(path: str) -> Dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path).items()}

def save_anchor(path: str, weights: Dict[str, np.ndarray]) -> None:
    np.savez(path, **weights)

def save_delta(anchor: Dict[str, np.ndarray],
               target: Dict[str, np.ndarray],
               delta_path: str,
               dtype=np.float16) -> None:
    delta = {}
    for k in target:
        # store small typed delta
        delta[k] = (target[k] - anchor[k]).astype(dtype)
    np.savez(delta_path, **delta)

def reconstruct(anchor_path: str, delta_path: str) -> Dict[str, np.ndarray]:
    anchor = load_npz(anchor_path)
    delta = load_npz(delta_path)
    rec = {}
    for k in anchor:
        if k in delta:
            rec[k] = anchor[k] + delta[k].astype(anchor[k].dtype)
        else:
            # fall back (if some tensors unchanged)
            rec[k] = anchor[k]
    return rec
