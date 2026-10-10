"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DESIGN_EBN0, EB_N0_MAX, EB_N0_MIN, EB_N0_STEP, MAX_FRAMES, MIN_ERRORS
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from run_exp1 import run_unit_tests
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n_list = [int(x) for x in os.environ.get("POLAR_EXP3_N_LIST", "256,512").split(",")]
    rate = 0.5
    max_iter = int(os.environ.get("POLAR_BP_MAX_ITER", "50"))
    eb_n0_range = np.arange(max(EB_N0_MIN, 1.0), EB_N0_MAX + 1e-9, EB_N0_STEP)

    for n in n_list:
        k = n // 2
        info_idx, _, _ = ga_construction(n, k, DESIGN_EBN0)
        frozen_bits = np.ones(n, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        r_sc = run_simulation(
            n,
            k,
            eb_n0_range,
            sc_d,
            "sc",
            MAX_FRAMES,
            MIN_ERRORS,
            frozen_bits=frozen_bits,
            info_indices=info_idx,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{n}_R0.5.csv")

        def scl_d(llr_ch):
            u, _ = SCLDecoder(n, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        r_scl = run_simulation(
            n,
            k,
            eb_n0_range,
            scl_d,
            "scl",
            MAX_FRAMES,
            MIN_ERRORS,
            frozen_bits=frozen_bits,
            info_indices=info_idx,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{n}_R0.5.csv")

        bp_decoder = BPDecoder(n, frozen_bits, max_iter=max_iter)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            n,
            k,
            eb_n0_range,
            bp_d,
            "bp",
            MAX_FRAMES,
            MIN_ERRORS,
            frozen_bits=frozen_bits,
            info_indices=info_idx,
        )
        all_results[f"BP (max_iter={max_iter})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{n}_R0.5.csv")

        shannon_db = find_capacity_limit(rate)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={n}, R={rate})",
            f"results/fig3_bp_N{n}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={n}, max_iter={max_iter})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{n}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{n}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
