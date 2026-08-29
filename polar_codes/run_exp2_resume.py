#!/usr/bin/env python3
"""从 L=4 起继续实验二（跳过已完成的 SC 与 SCL L=2）。"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_REMAINING = [4, 8]
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(5.0, 10.01, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

all_results = {}
all_results["SC (L=1)"] = load_results_csv("results/exp2_sc_N512_R0.5.csv")
all_results["SCL (L=2)"] = load_results_csv("results/exp2_scl_L2_N512_R0.5.csv")

for L in L_REMAINING:
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")
    scl = SCLDecoder(N, frozen_bits, list_size=L, crc_length=0)

    def scl_decoder(llr_ch, _scl=scl):
        return _scl.decode(llr_ch)[0], None

    results = run_simulation(
        N, K, EB_N0_RANGE, scl_decoder, "scl",
        MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")
cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH)

def cascl_decoder(llr_ch, _cascl=cascl):
    return _cascl.decode(llr_ch)[0], None

results_cascl = run_simulation(
    N, K, EB_N0_RANGE, cascl_decoder, "scl",
    MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
    info_indices=info_idx, verbose=True,
)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    f"SCL vs SC BLER (N={N}, R={RATE})",
    "results/fig2_scl_bler.png",
    shannon_limit_db=shannon_db,
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
print("\n实验二续跑完成。")
