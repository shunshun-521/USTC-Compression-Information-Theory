#!/usr/bin/env python3
"""生成 exp2 图表与 CA-SCL CSV（快速补全）。"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

N = 512
K = N // 2
RATE = 0.5
CRC_LENGTH = 8
EB_N0_RANGE = np.arange(5.0, 9.76, 0.25)  # 高 SNR 点单独限制帧数

info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

if not os.path.exists("results/exp2_cascl_L8_N512_R0.5.csv"):
    print("运行 CA-SCL 快速仿真...")
    cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH)
    results_cascl = run_simulation(
        N, K, EB_N0_RANGE,
        lambda llr, c=cascl: (c.decode(llr)[0], None),
        "scl", 20000, 100, crc_length=CRC_LENGTH, info_indices=info_idx,
    )
    save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

all_results = {
    "SC (L=1)": load_results_csv("results/exp2_sc_N512_R0.5.csv"),
    "SCL (L=2)": load_results_csv("results/exp2_scl_L2_N512_R0.5.csv"),
    "SCL (L=4)": load_results_csv("results/exp2_scl_L4_N512_R0.5.csv"),
    "SCL (L=8)": load_results_csv("results/exp2_scl_L8_N512_R0.5.csv"),
    f"CA-SCL (L=8, CRC={CRC_LENGTH})": load_results_csv(
        "results/exp2_cascl_L8_N512_R0.5.csv"
    ),
}

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results, f"SCL vs SC BLER (N={N}, R={RATE})",
    "results/fig2_scl_bler.png", shannon_limit_db=shannon_db,
)

labels = list(all_results.keys())
avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, avg_times)
ax.set_xlabel("Decoder")
ax.set_ylabel("Avg Decode Time (ms)")
ax.set_title(f"Decoding Time vs List Size (N={N})")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("results/fig2_decode_time.png", dpi=150)
plt.savefig("results/fig2_decode_time.pdf")
plt.close()
print("fig2 生成完成。")
