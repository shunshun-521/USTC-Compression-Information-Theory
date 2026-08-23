#!/usr/bin/env python3
"""以较低帧数运行全部实验（用于快速生成结果）。"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("results", exist_ok=True)

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv
from run_exp1 import run_unit_tests

MAX_FRAMES = 2000
MIN_ERRORS = 25
DESIGN_EBN0 = 2.5
RATE = 0.5

if __name__ == "__main__":
    run_unit_tests()

    # Exp2
    N = 512
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    eb2 = np.arange(1.0, 5.5, 0.5)
    all2 = {}

    def sc_d(llr):
        return sc_decode(llr, frozen_bool), None

    print("Exp2 SC")
    r = run_simulation(N, K, eb2, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all2["SC (L=1)"] = r
    save_results_csv(r, "results/exp2_sc_N512_R0.5.csv")

    for L in [2, 4, 8]:
        print(f"Exp2 SCL L={L}")

        def scl_d(llr, _L=L):
            u, _ = SCLDecoder(N, frozen_bool, list_size=_L).decode(llr)
            return u, None

        r = run_simulation(N, K, eb2, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all2[f"SCL (L={L})"] = r
        save_results_csv(r, f"results/exp2_scl_L{L}_N512_R0.5.csv")

    print("Exp2 CA-SCL")

    def cascl_d(llr):
        u, _ = SCLDecoder(N, frozen_bool, 8, crc_length=8).decode(llr)
        return u, None

    r = run_simulation(N, K, eb2, cascl_d, "scl", MAX_FRAMES, MIN_ERRORS,
                       crc_length=8, info_indices=info_idx)
    all2["CA-SCL (L=8, CRC=8)"] = r
    save_results_csv(r, "results/exp2_cascl_L8_N512_R0.5.csv")
    save_results_csv(r, "results/exp2_scl_N512_R0.5.csv")

    shannon = find_capacity_limit(RATE)
    plot_bler_curves(all2, f"SCL vs SC (N={N})", "results/fig2_scl_bler.png", shannon)
    labels = list(all2.keys())
    times = [np.mean([x["avg_decode_time"] for x in v]) * 1000 for v in all2.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), times)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Avg Decode Time (ms)")
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()

    # Exp3
    eb3 = np.arange(1.0, 5.5, 0.5)
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bool = np.ones(N, dtype=bool)
        frozen_bool[info_idx] = False
        all3 = {}

        r = run_simulation(N, K, eb3, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all3["SC"] = r
        save_results_csv(r, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl4(llr):
            u, _ = SCLDecoder(N, frozen_bool, 4).decode(llr)
            return u, None

        r = run_simulation(N, K, eb3, scl4, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all3["SCL (L=4)"] = r
        save_results_csv(r, f"results/exp3_scl_N{N}_R0.5.csv")

        bp = BPDecoder(N, frozen_bool, 50)

        def bp_d(llr):
            u, it = bp.decode(llr)
            return u, it

        r = run_simulation(N, K, eb3, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all3["BP (max_iter=50)"] = r
        save_results_csv(r, f"results/exp3_bp_N{N}_R0.5.csv")

        plot_bler_curves(all3, f"SC vs SCL vs BP (N={N})", f"results/fig3_bp_N{N}_bler.png", shannon)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x["eb_n0_db"] for x in r], [x["avg_iters"] for x in r], "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("All reduced experiments completed.")
