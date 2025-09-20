# utils.py

import os, time, csv, hashlib, torch

def hash_stage(config: dict, stage_idx: int) -> str:
    """
    Create a unique hash for a stage based on trial config + stage index.
    """
    key = str(config) + f"_stage{stage_idx}"
    return hashlib.sha256(key.encode()).hexdigest()

import os, csv

class CSVLogger:
    def __init__(self, filepath="stage_log.csv", fieldnames=None):
        self.filepath = filepath
        self.fieldnames = fieldnames or [
            "trial", "stage_idx", "iters", "lr", "runtime",
            "cache_hit", "cache_miss", "adapter_path"
        ]
        # Write header only if the file doesn't exist or is empty
        needs_header = (not os.path.exists(self.filepath)) or (os.path.getsize(self.filepath) == 0)
        if needs_header:
            with open(self.filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, **row):
        # Ensure every row has the same schema
        payload = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.filepath, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(payload)


def timed(func):
    """
    Decorator to measure function runtime.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper
