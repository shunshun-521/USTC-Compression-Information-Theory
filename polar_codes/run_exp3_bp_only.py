"""快速完成实验三 BP 部分（降低高信噪比点仿真量）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 25
MAX_FRAMES = 15000
MIN_ERRORS = 30
EB_N0_RANGE = np.arange(1.0, 4.25, 0.25)

if __name__ == "__main__":
    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0, RATE)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)
        print(f"BP N={N}")
        r_bp = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            lambda llr: bp_decoder.decode(llr),
            "bp",
            MAX_FRAMES,
            MIN_ERRORS,
            info_indices=info_idx,
        )
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        all_results = {}
        sc_path = f"results/exp3_sc_N{N}_R0.5.csv"
        scl_path = f"results/exp3_scl_N{N}_R0.5.csv"
        if os.path.exists(sc_path):
            all_results["SC"] = load_results_csv(sc_path)
        if os.path.exists(scl_path):
            all_results["SCL (L=4)"] = load_results_csv(scl_path)
        all_results[f"BP (max_iter={MAX_ITER})"] = r_bp

        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("实验三 BP 完成。")
