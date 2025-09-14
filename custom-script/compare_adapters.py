import numpy as np

a1 = np.load("adapters_lr1e-5.npz")
a2 = np.load("adapters_lr5e-5.npz")

print("Keys:", len(a1.files))

for key in a1.files:
    if key in a2:
        diff = np.mean(np.abs(a1[key] - a2[key]))
        print(f"{key}: mean difference {diff:.6f}")