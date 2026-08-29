"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import fast_sim_env, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def main():
    from validate import main as validate_main

    validate_main()

    os.makedirs("results", exist_ok=True)

    N_LIST = [256, 512]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_ITER = 50
    MAX_FRAMES = 1500 if fast_sim_env() else 100000
    MIN_ERRORS = 20 if fast_sim_env() else 100
    EB_N0_RANGE = (
        np.arange(2.0, 5.5, 0.5)
        if fast_sim_env()
        else np.arange(1.0, 5.5, 0.25)
    )

    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        print(f"\n{'=' * 60}")
        print(f"实验三: N={N}, K={K}")
        print(f"{'=' * 60}")

        r_sc = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            lambda llr: (sc_decode(llr, frozen_bits), None),
            "sc",
            MAX_FRAMES,
            MIN_ERRORS,
            info_indices=info_idx,
            frozen_bits=frozen_bits,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        r_scl = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            lambda llr: (
                SCLDecoder(N, frozen_bits, list_size=4).decode(llr)[0],
                None,
            ),
            "scl",
            MAX_FRAMES,
            MIN_ERRORS,
            info_indices=info_idx,
            frozen_bits=frozen_bits,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

        def bp_d(llr_ch, _bp=bp_decoder):
            u_hat, num_iters = _bp.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            bp_d,
            "bp",
            MAX_FRAMES,
            MIN_ERRORS,
            info_indices=info_idx,
            frozen_bits=frozen_bits,
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
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
