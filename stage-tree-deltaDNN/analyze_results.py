# analyze_results.py
import pandas as pd
from utils import fmt_bytes

df = pd.read_csv("stage_log.csv")

# Basic cleanup/coercion
for col in ["stage_idx","iters","runtime_sec","cache_hit","cache_miss",
            "size_full_bytes","size_delta_bytes","compression_ratio_full_over_delta","lr",
            "fastcdc_chunks"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 1) Per-trial stage table (like HiPPO)
print("\n=== Per-trial stages ===")
show = df[["trial","stage_idx","iters","lr","runtime_sec","cache_hit",
           "size_full_bytes","size_delta_bytes","compression_ratio_full_over_delta"]].copy()
# Prettify sizes
show["size_full"]  = show["size_full_bytes"].apply(lambda x: fmt_bytes(int(x)) if pd.notna(x) else "")
show["size_delta"] = show["size_delta_bytes"].apply(lambda x: fmt_bytes(int(x)) if pd.notna(x) else "")
show = show.drop(columns=["size_full_bytes","size_delta_bytes"])
print(show.to_string(index=False))

# 2) Compute reuse: compare iters without reuse vs with reuse
iters_built = df.loc[df["cache_hit"]==0, "iters"].sum()
iters_reused = df.loc[df["cache_hit"]==1, "iters"].sum()
iters_no_reuse = iters_built + iters_reused
iters_with_reuse = iters_built
saved_iters = iters_no_reuse - iters_with_reuse
pct_saved = (saved_iters / iters_no_reuse * 100.0) if iters_no_reuse > 0 else 0.0
speedup = (iters_no_reuse / iters_with_reuse) if iters_with_reuse > 0 else 1.0

print("\n=== Compute reuse summary ===")
print(f"Iterations (no reuse baseline): {iters_no_reuse}")
print(f"Iterations (with reuse):        {iters_with_reuse}")
print(f"Saved iterations:               {saved_iters}  ({pct_saved:.2f}%)")
print(f"Speedup:                        {speedup:.2f}×")

# 3) Storage: anchor vs delta
print("\n=== Storage savings (DeltaDNN-style) ===")
st2 = df[df["stage_idx"]>1].copy()
if len(st2) > 0:
    total_full = st2["size_full_bytes"].sum()
    total_delta = st2["size_delta_bytes"].sum()
    ratio = (total_full / total_delta) if total_delta > 0 else float("inf")
    print(f"Total full size (stages>1):  {fmt_bytes(int(total_full))}")
    print(f"Total delta size (stages>1): {fmt_bytes(int(total_delta))}")
    print(f"Aggregate compression ratio: {ratio:.2f}×")
else:
    print("No stages >1 found; nothing to compress.")

# 4) Optional: FastCDC chunks per produced file
if "fastcdc_chunks" in df.columns and df["fastcdc_chunks"].notna().any():
    print("\n=== FastCDC chunk counts (optional) ===")
    print(df[["trial","stage_idx","fastcdc_chunks","adapter_path"]].to_string(index=False))
