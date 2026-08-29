#!/usr/bin/env python3
"""快速完成实验二/三剩余部分（高 SNR 点限制 max_frames）。"""
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
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import (
    save_results_csv,
    plot_bler_curves,
    find_capacity_limit,
    load_results_csv,
)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
MAX_FRAMES = 30000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(5.0, 10.01, 0.25)

info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0

# --- 完成 exp2 L=8 与 CA-SCL ---
all_results = {
    "SC (L=1)": load_results_csv("results/exp2_sc_N512_R0.5.csv"),
    "SCL (L=2)": load_results_csv("results/exp2_scl_L2_N512_R0.5.csv"),
    "SCL (L=4)": load_results_csv("results/exp2_scl_L4_N512_R0.5.csv"),
}

for L in [8]:
    print(f"\nSCL L={L}")
    scl = SCLDecoder(N, frozen_bits, list_size=L)
    results = run_simulation(
        N, K, EB_N0_RANGE,
        lambda llr, s=scl: (s.decode(llr)[0], None),
        "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
    )
    all_results[f"SCL (L={L})"] = results
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

print("\nCA-SCL L=8")
cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH)
results_cascl = run_simulation(
    N, K, EB_N0_RANGE,
    lambda llr, c=cascl: (c.decode(llr)[0], None),
    "scl", MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH, info_indices=info_idx,
)
all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

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

# --- exp3 ---
for N3 in [256, 512]:
    K3 = N3 // 2
    info3, _, _ = ga_construction(N3, K3, DESIGN_EBN0)
    fb3 = np.ones(N3, dtype=int)
    fb3[info3] = 0
    all3 = {}

    results_sc = run_simulation(
        N3, K3, EB_N0_RANGE,
        lambda llr, fb=fb3: (sc_decode(llr, fb), None),
        "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info3,
    )
    all3["SC"] = results_sc
    save_results_csv(results_sc, f"results/exp3_sc_N{N3}_R0.5.csv")

    scl3 = SCLDecoder(N3, fb3, list_size=4)
    results_scl = run_simulation(
        N3, K3, EB_N0_RANGE,
        lambda llr, s=scl3: (s.decode(llr)[0], None),
        "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info3,
    )
    all3["SCL (L=4)"] = results_scl
    save_results_csv(results_scl, f"results/exp3_scl_N{N3}_R0.5.csv")

    bp3 = BPDecoder(N3, fb3, max_iter=50)
    results_bp = run_simulation(
        N3, K3, EB_N0_RANGE,
        lambda llr, b=bp3: b.decode(llr),
        "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info3,
    )
    all3["BP (max_iter=50)"] = results_bp
    save_results_csv(results_bp, f"results/exp3_bp_N{N3}_R0.5.csv")

    plot_bler_curves(
        all3, f"SC vs SCL vs BP (N={N3}, R={RATE})",
        f"results/fig3_bp_N{N3}_bler.png", shannon_limit_db=shannon_db,
    )
    eb = [r["eb_n0_db"] for r in results_bp]
    iters = [r["avg_iters"] for r in results_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, iters, "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N3}, max_iter=50)")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N3}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N3}_iters.pdf")
    plt.close()
    print(f"exp3 N={N3} done")

print("\n全部实验完成。")
