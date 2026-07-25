#!/usr/bin/env python3
"""重新生成实验一、二的图表（仿真数据已存在时）。"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_results_csv, plot_bler_curves, find_capacity_limit

RATE = 0.5

# 实验一图
all_results = {}
for N in [256, 512, 1024]:
    path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.path.exists(path):
        all_results[f"SC, N={N}, K={N//2}"] = load_results_csv(path)

if all_results:
    plot_bler_curves(
        all_results,
        f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        "results/fig1_sc_bler.png",
        shannon_limit_db=find_capacity_limit(RATE),
    )

# 实验二图
all_results2 = {}
mapping = {
    "SC (L=1)": "results/exp2_sc_N512_R0.5.csv",
    "SCL (L=2)": "results/exp2_scl_L2_N512_R0.5.csv",
    "SCL (L=4)": "results/exp2_scl_L4_N512_R0.5.csv",
    "SCL (L=8)": "results/exp2_scl_L8_N512_R0.5.csv",
    "CA-SCL (L=8, CRC=8)": "results/exp2_cascl_L8_N512_R0.5.csv",
}
for label, path in mapping.items():
    if os.path.exists(path):
        all_results2[label] = load_results_csv(path)

if all_results2:
    plot_bler_curves(
        all_results2,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
        shannon_limit_db=find_capacity_limit(RATE),
    )
    labels = list(all_results2.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results2.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()

print("图表已更新。")
