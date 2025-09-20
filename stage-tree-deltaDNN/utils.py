# utils.py
import os, csv, time, json
from dataclasses import asdict
from typing import Dict, Any, Iterable

class CSVLogger:
    def __init__(self, path: str = "stage_log.csv"):
        self.path = path
        self._has_header = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, row: Dict[str, Any]) -> None:
        # Flatten any dataclasses if passed
        flat = {}
        for k, v in row.items():
            if hasattr(v, "__dataclass_fields__"):
                flat[k] = json.dumps(asdict(v), sort_keys=True)
            else:
                flat[k] = v

        keys = list(flat.keys())
        write_header = not self._has_header
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
                self._has_header = True
            w.writerow(flat)

def sizeof(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0

def fmt_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
