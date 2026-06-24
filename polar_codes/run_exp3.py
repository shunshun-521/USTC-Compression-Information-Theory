"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 与 SC、SCL（L=4）对比
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from tests_unit import run_all as run_unit_tests
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)

QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

N_LIST = [256] if QUICK else [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 500 if QUICK else 100000
MIN_ERRORS = 10 if QUICK else 100
EB_N0_RANGE = np.arange(2.0, 4.0, 0.5) if QUICK else np.arange(1.0, 5.5, 0.25)


def main():
    print("运行单元测试...")
    run_unit_tests()

    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\nN={N} SC 仿真")
        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, verbose=True,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr_ch):
            u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        print(f"\nN={N} SCL L=4 仿真")
        r_scl = run_simulation(
            N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, verbose=True,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"\nN={N} BP 仿真")
        r_bp = run_simulation(
            N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, verbose=True,
        )
        all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

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
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
