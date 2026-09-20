"""从已有 CSV 生成缺失的图表。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RATE = 0.5
shannon_db = find_capacity_limit(RATE)
print(f"Shannon limit R={RATE}: {shannon_db:.3f} dB")

# exp1 figures
exp1 = {}
for N in [256, 512, 1024]:
    path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.path.exists(path):
        exp1[f"SC, N={N}, K={N//2}"] = load_results_csv(path)
if exp1:
    plot_bler_curves(exp1, f"SC Decoder BLER vs Eb/N0 (R={RATE})", "results/fig1_sc_bler.png", shannon_db)

# exp2 figures
exp2 = {}
mapping = {
    "SC (L=1)": "results/exp2_sc_N512_R0.5.csv",
    "SCL (L=2)": "results/exp2_scl_L2_N512_R0.5.csv",
    "SCL (L=4)": "results/exp2_scl_L4_N512_R0.5.csv",
    "SCL (L=8)": "results/exp2_scl_L8_N512_R0.5.csv",
}
for label, path in mapping.items():
    if os.path.exists(path):
        exp2[label] = load_results_csv(path)
if exp2:
    plot_bler_curves(exp2, "SCL vs SC BLER (N=512, R=0.5)", "results/fig2_scl_bler.png", shannon_db)

print("Plots regenerated.")
