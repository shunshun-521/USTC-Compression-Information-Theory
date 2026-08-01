#!/usr/bin/env python3
"""快速完成剩余实验（缩减帧数）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs("results", exist_ok=True)
MAX_F, MIN_E = 2000, 25
EB = np.arange(6.0, 12.5, 0.5)
RATE = 0.5

# Exp2: L8 + CA-SCL
N, K = 512, 256
info, _, _ = ga_construction(N, K, 2.5)
frozen = np.ones(N, dtype=bool)
frozen[info] = False

for L, crc, name, path in [
    (8, 0, "SCL (L=8)", "results/exp2_scl_L8_N512_R0.5.csv"),
    (8, 8, "CA-SCL", "results/exp2_cascl_L8_N512_R0.5.csv"),
]:
    print(f"Running {name}...")

    def dec(llr, _L=L, _crc=crc):
        u, _ = SCLDecoder(N, frozen, _L, _crc).decode(llr)
        return u, None

    r = run_simulation(
        N, K, EB, dec, "scl", MAX_F, MIN_E,
        crc_length=crc, info_indices=info, verbose=True,
    )
    save_results_csv(r, path)

# Exp3: N=256 and N=512
for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    all_results = {}

    def sc_d(llr):
        return sc_decode(llr, frozen), None

    print(f"Exp3 N={N} SC")
    r_sc = run_simulation(N, K, EB, sc_d, "sc", MAX_F, MIN_E, info_indices=info)
    save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")
    all_results["SC"] = r_sc

    def scl_d(llr):
        u, _ = SCLDecoder(N, frozen, 4).decode(llr)
        return u, None

    print(f"Exp3 N={N} SCL")
    r_scl = run_simulation(N, K, EB, scl_d, "scl", MAX_F, MIN_E, info_indices=info)
    save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")
    all_results["SCL (L=4)"] = r_scl

    bp = BPDecoder(N, frozen, max_iter=50)

    def bp_d(llr):
        u, it = bp.decode(llr)
        return u, it

    print(f"Exp3 N={N} BP")
    r_bp = run_simulation(
        N, K, EB, bp_d, "bp",
        1000 if N >= 512 else MAX_F,
        20, info_indices=info, verbose=True,
    )
    save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")
    all_results["BP (max_iter=50)"] = r_bp

    shannon = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        f"results/fig3_bp_N{N}_bler.png",
        shannon_limit_db=shannon,
    )

    eb_vals = [x["eb_n0_db"] for x in r_bp]
    iters = [x["avg_iters"] for x in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb_vals, iters, "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

print("All remaining experiments done.")
