#!/usr/bin/env python3
"""生成实验二完整图表。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_results_csv, plot_bler_curves, find_capacity_limit

RESULTS = "results"
RATE = 0.5
N = 512

all_results = {}
mapping = [
    ("SC (L=1)", f"{RESULTS}/exp2_sc_N{N}_R0.5.csv"),
    ("SCL (L=2)", f"{RESULTS}/exp2_scl_L2_N{N}_R0.5.csv"),
    ("SCL (L=4)", f"{RESULTS}/exp2_scl_L4_N{N}_R0.5.csv"),
    ("SCL (L=8)", f"{RESULTS}/exp2_scl_L8_N{N}_R0.5.csv"),
    ("CA-SCL (L=8, CRC=8)", f"{RESULTS}/exp2_cascl_L8_N{N}_R0.5.csv"),
]
for label, path in mapping:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R={RATE})",
    f"{RESULTS}/fig2_scl_bler.png",
    shannon_limit_db=shannon_db,
)

labels = list(all_results.keys())
avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels, avg_times)
ax.set_xlabel("Decoder")
ax.set_ylabel("Avg Decode Time (ms)")
ax.set_title(f"Decoding Time vs List Size (N={N})")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig2_decode_time.png", dpi=150)
plt.savefig(f"{RESULTS}/fig2_decode_time.pdf")
plt.close()
print("fig2 updated with", len(all_results), "curves")
