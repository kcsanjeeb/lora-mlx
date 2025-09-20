import os, json, hashlib, shutil, subprocess, time
from typing import List, Any, Optional
from dataclasses import dataclass, asdict
from utils import CSVLogger

# --------- Utilities ---------

def run_command_with_live_output(cmd: List[str]) -> int:
    """Stream a subprocess' stdout live (your existing helper is fine too)."""
    print(">>", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(proc.stdout.readline, ""):
        if line:
            print(line, end="")
    proc.stdout.close()
    return proc.wait()

def sha256_of_obj(obj: Any) -> str:
    """Stable hash for dict/list of simple types."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]  # short for readability

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# --------- Config dataclasses ---------

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

# --------- Stage Tree Runner ---------

class StageTreeRunner:
    """
    HiPPO-style stage runner that reuses shared prefixes across trials.
    Logs stage runtimes, cache hits/misses, and metrics to CSV.
    """
    def __init__(self, static: StaticCfg,
                 cache_dir: str = "./stage_cache",
                 lora_script: str = "./scripts/lora.py",   # <— FIXED
                 log_file: str = "stage_log.csv"):
        self.static = static
        self.cache_dir = cache_dir
        self.lora_script = os.path.abspath(lora_script)  # resolve to full path
        self.logger = CSVLogger(log_file)
        ensure_dir(self.cache_dir)

    # ---- Keys & paths ----
    def prefix_key(self, stages_prefix: List[Stage]) -> str:
        key_obj = {
            "static": asdict(self.static),
            "prefix_stages": [asdict(s) for s in stages_prefix],
        }
        return sha256_of_obj(key_obj)

    def prefix_dir(self, prefix_key: str) -> str:
        return os.path.join(self.cache_dir, prefix_key)

    def prefix_adapter_path(self, prefix_key: str) -> str:
        return os.path.join(self.prefix_dir(prefix_key), "adapters.npz")

    def has_prefix_ckpt(self, prefix_key: str) -> bool:
        return os.path.exists(self.prefix_adapter_path(prefix_key))

    # ---- Training a single stage (continues from prev adapters if present) ----
    def _train_stage(self, stage: Stage, starting_adapters: Optional[str], out_adapter: str) -> str:
        cmd = [
            "python", self.lora_script,
            "--model", self.static.model_path,
            "--train",
            "--iters", str(stage.iters),
            "--steps-per-eval", str(self.static.steps_per_eval),
            "--val-batches", str(self.static.val_batches),
            "--learning-rate", str(stage.learning_rate),
            "--lora-layers", str(self.static.lora_layers),
            "--adapter-file", out_adapter,  # force output file
        ]
        if starting_adapters is not None:
            cmd += ["--resume-adapter-file", starting_adapters]  # resume from prev stage

        rc = run_command_with_live_output(cmd)
        if rc != 0:
            raise RuntimeError(f"Training stage failed with return code {rc}")

        if not os.path.exists(out_adapter):
            raise FileNotFoundError(f"Expected adapter file after training, not found: {out_adapter}")

        return out_adapter

    # ---- Build (or reuse) up to a given prefix length ----
    def ensure_prefix(self, trial: Trial, prefix_len: int) -> str:
        assert 1 <= prefix_len <= len(trial.stages)
        prev_adapter_path = None
        for i in range(1, prefix_len + 1):
            stage = trial.stages[i - 1]
            key = self.prefix_key(trial.stages[:i])
            out_dir = self.prefix_dir(key)
            out_adapter = self.prefix_adapter_path(key)

            if self.has_prefix_ckpt(key):
                print(f"[cache hit] {trial.name} prefix {i}/{len(trial.stages)} → {out_adapter}")
                self.logger.log({
                    "trial": trial.name,
                    "stage_idx": i,
                    "iters": stage.iters,
                    "lr": stage.learning_rate,
                    "runtime": 0.0,
                    "cache_hit": 1,
                    "cache_miss": 0,
                    "adapter_path": out_adapter,
                })
                prev_adapter_path = out_adapter
                continue

            print(f"[build] {trial.name} stage {i}/{len(trial.stages)} (iters={stage.iters}, lr={stage.learning_rate})")
            ensure_dir(out_dir)

            start = time.time()
            produced = self._train_stage(stage, starting_adapters=prev_adapter_path, out_adapter=out_adapter)
            runtime = time.time() - start

            self.logger.log({
                "trial": trial.name,
                "stage_idx": i,
                "iters": stage.iters,
                "lr": stage.learning_rate,
                "runtime": runtime,
                "cache_hit": 0,
                "cache_miss": 1,
                "adapter_path": produced,
            })

            print(f"[saved] {produced}")
            prev_adapter_path = produced

        return prev_adapter_path

    def run_trial(self, trial: Trial) -> str:
        final_path = self.ensure_prefix(trial, prefix_len=len(trial.stages))
        print(f"[done] Trial {trial.name} final adapters at: {final_path}")
        return final_path
