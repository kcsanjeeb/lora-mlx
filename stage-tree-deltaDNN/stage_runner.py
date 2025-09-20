# stage_runner.py
import os, json, hashlib, shutil, subprocess, time
from typing import List, Any, Optional, Dict
from dataclasses import dataclass, asdict

from utils import CSVLogger, sizeof, now_iso
from delta_store import load_npz, save_anchor, save_delta

# ---------- small helpers ----------
def run_command_stream(cmd: List[str]) -> int:
    print(">>", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(proc.stdout.readline, ""):
        if line:
            print(line, end="")
    proc.stdout.close()
    return proc.wait()

def sha256_of_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ---------- config ----------
@dataclass(frozen=True)
class StaticCfg:
    model_path: str
    lora_layers: int
    val_batches: int
    steps_per_eval: int
    seq_len: Optional[int] = None
    dataset_id: Optional[str] = None
    tokenizer_id: Optional[str] = None

@dataclass(frozen=True)
class Stage:
    iters: int
    learning_rate: float

@dataclass(frozen=True)
class Trial:
    name: str
    stages: List[Stage]

# ---------- runner ----------
class StageTreeRunner:
    """
    HiPPO-style stage runner with DeltaDNN-style storage.
    - Compute reuse: cache prefix checkpoints by content hash
    - Storage savings: stage 1 stored as anchor; later stages store delta vs. anchor
    - Metrics logged to CSV per stage: runtime, cache_hit, sizes, compression_ratio
    """
    def __init__(self,
                 static: StaticCfg,
                 cache_dir: str = "./stage_cache",
                 lora_script: str = "./scripts/lora.py",
                 log_file: str = "stage_log.csv",
                 fastcdc_bin: Optional[str] = None,  # optional
                 ):
        self.static = static
        self.cache_dir = cache_dir
        self.lora_script = os.path.abspath(lora_script)
        self.logger = CSVLogger(log_file)
        self.fastcdc_bin = fastcdc_bin  # if provided, we’ll log chunk count per file
        ensure_dir(self.cache_dir)

    # ---- keys & paths
    def prefix_key(self, stages_prefix: List[Stage]) -> str:
        key_obj = {
            "static": asdict(self.static),
            "prefix_stages": [asdict(s) for s in stages_prefix],
        }
        return sha256_of_obj(key_obj)

    def prefix_dir(self, key: str) -> str:
        return os.path.join(self.cache_dir, key)

    def adapter_path(self, key: str) -> str:
        return os.path.join(self.prefix_dir(key), "adapters.npz")

    def delta_path(self, key: str) -> str:
        return os.path.join(self.prefix_dir(key), "adapters_delta.npz")

    def has_ckpt(self, key: str) -> bool:
        return os.path.exists(self.adapter_path(key))

    # ---- training a single stage
    def _train_stage(self, stage: Stage, out_adapter: str, resume_from: Optional[str]) -> None:
        cmd = [
            "python", self.lora_script,
            "--model", self.static.model_path,
            "--train",
            "--iters", str(stage.iters),
            "--steps-per-eval", str(self.static.steps_per_eval),
            "--val-batches", str(self.static.val_batches),
            "--learning-rate", str(stage.learning_rate),
            "--lora-layers", str(self.static.lora_layers),
            "--adapter-file", out_adapter,
        ]
        if resume_from:
            cmd += ["--resume-adapter-file", resume_from]

        rc = run_command_stream(cmd)
        if rc != 0:
            raise RuntimeError(f"Training failed (rc={rc}).")
        if not os.path.exists(out_adapter):
            raise FileNotFoundError(f"Expected {out_adapter} after training.")

    # ---- optional: FastCDC (chunk count only; comparison done in analysis)
    def _fastcdc_chunk_count(self, path: str) -> Optional[int]:
        if self.fastcdc_bin is None or not os.path.exists(self.fastcdc_bin):
            return None
        try:
            # Use moderate params; output is "offset size hash"
            proc = subprocess.run(
                [self.fastcdc_bin, "-file", path, "-min", "4096", "-avg", "8192", "-max", "16384"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            )
            if proc.returncode != 0:
                return None
            # Count lines
            return sum(1 for _ in proc.stdout.splitlines() if _.strip())
        except Exception:
            return None

    # ---- ensure prefix (1..prefix_len)
    def ensure_prefix(self, trial: Trial, prefix_len: int) -> str:
        assert 1 <= prefix_len <= len(trial.stages)
        anchor_key: Optional[str] = None
        prev_key: Optional[str] = None

        for i in range(1, prefix_len + 1):
            prefix = trial.stages[:i]
            key = self.prefix_key(prefix)
            out_dir = self.prefix_dir(key)
            out_adapter = self.adapter_path(key)

            ensure_dir(out_dir)

            if self.has_ckpt(key):
                # cache hit (compute reuse)
                self.logger.log({
                    "trial": trial.name,
                    "stage_idx": i,
                    "iters": prefix[-1].iters,
                    "lr": prefix[-1].learning_rate,
                    "runtime_sec": 0.0,
                    "cache_hit": 1,
                    "cache_miss": 0,
                    "adapter_path": out_adapter,
                    "size_full_bytes": sizeof(out_adapter),
                    "size_delta_bytes": sizeof(self.delta_path(key)) if i > 1 else 0,
                    "compression_ratio_full_over_delta": (sizeof(out_adapter)/max(sizeof(self.delta_path(key)),1)) if i > 1 else 1.0,
                    "prefix_key": key,
                    "wall_start": now_iso(),
                    "wall_end": now_iso(),
                    "fastcdc_chunks": self._fastcdc_chunk_count(out_adapter),
                })
                prev_key = key
                if i == 1:
                    anchor_key = key
                continue

            # build this stage
            stage_obj = prefix[-1]
            start = time.time()
            print(f"[build] {trial.name} stage {i}/{len(trial.stages)} (iters={stage_obj.iters}, lr={stage_obj.learning_rate})")

            resume_from = self.adapter_path(prev_key) if prev_key else None
            self._train_stage(stage_obj, out_adapter, resume_from)
            runtime = time.time() - start

            # DeltaDNN storage:
            if i == 1:
                # Save as anchor (full)
                weights = load_npz(out_adapter)
                save_anchor(out_adapter, weights)  # this just ensures canonical npz write
                anchor_key = key
                size_delta = 0
                cr = 1.0
            else:
                # Store delta vs anchor (stage 1)
                assert anchor_key is not None, "Anchor must exist (stage 1)."
                anchor_path = self.adapter_path(anchor_key)
                delta_out = self.delta_path(key)

                anchor_w = load_npz(anchor_path)
                target_w = load_npz(out_adapter)
                save_delta(anchor_w, target_w, delta_out)

                size_delta = sizeof(delta_out)
                cr = (sizeof(out_adapter) / max(size_delta, 1))

            # log
            self.logger.log({
                "trial": trial.name,
                "stage_idx": i,
                "iters": stage_obj.iters,
                "lr": stage_obj.learning_rate,
                "runtime_sec": round(runtime, 3),
                "cache_hit": 0,
                "cache_miss": 1,
                "adapter_path": out_adapter,
                "size_full_bytes": sizeof(out_adapter),
                "size_delta_bytes": size_delta,
                "compression_ratio_full_over_delta": cr,
                "prefix_key": key,
                "wall_start": now_iso(),
                "wall_end": now_iso(),
                "fastcdc_chunks": self._fastcdc_chunk_count(out_adapter),
            })

            prev_key = key

        return self.adapter_path(prev_key)

    def run_trial(self, trial: Trial) -> str:
        final_path = self.ensure_prefix(trial, prefix_len=len(trial.stages))
        print(f"[done] Trial {trial.name} final adapters at: {final_path}")
        return final_path
