"""仅重跑实验三 BP 部分并更新对比图（复用已有 SC/SCL CSV）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction_for_simulation
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)


def main():
    os.makedirs("results", exist_ok=True)
    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction_for_simulation(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        bp_decoder = BPDecoder(N, frozen_bits.astype(bool), max_iter=MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"\n{'='*60}\nBP 重跑 N={N}\n{'='*60}")
        r_bp = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            bp_d,
            "bp",
            MAX_FRAMES,
            MIN_ERRORS,
            info_indices=info_idx,
        )
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        all_results = {}
        sc_path = f"results/exp3_sc_N{N}_R0.5.csv"
        scl_path = f"results/exp3_scl_N{N}_R0.5.csv"
        if os.path.isfile(sc_path):
            all_results["SC"] = load_results_csv(sc_path)
        if os.path.isfile(scl_path):
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
        print(f"N={N} BP 完成。")

    with open("results/ALL_DONE.txt", "w") as f:
        f.write("DONE\n")
    print("\nBP 重跑全部完成。")


if __name__ == "__main__":
    main()
