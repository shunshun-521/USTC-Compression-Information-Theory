# run_exp3.py
"""实验三：BP 译码"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    quick = os.environ.get("POLAR_QUICK", "0") == "1"

    n_list = [256, 512] if not quick else [256]
    rate = 0.5
    design_ebn0 = 2.5
    max_iter = 50
    max_frames = 1500 if quick else 100000
    min_errors = 15 if quick else 100
    eb_n0_range = (
        np.arange(2.0, 4.5, 0.5) if quick else np.arange(1.0, 5.5, 0.25)
    )

    for N in n_list:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, design_ebn0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        r_sc = run_simulation(
            N,
            K,
            eb_n0_range,
            sc_d,
            "sc",
            max_frames,
            min_errors,
            info_indices=info_idx,
            verbose=True,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, os.path.join(RESULTS_DIR, f"exp3_sc_N{N}_R0.5.csv"))

        def scl_d(llr_ch):
            u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        r_scl = run_simulation(
            N,
            K,
            eb_n0_range,
            scl_d,
            "scl",
            max_frames,
            min_errors,
            info_indices=info_idx,
            verbose=True,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, os.path.join(RESULTS_DIR, f"exp3_scl_N{N}_R0.5.csv"))

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=max_iter)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            N,
            K,
            eb_n0_range,
            bp_d,
            "bp",
            max_frames,
            min_errors,
            info_indices=info_idx,
            verbose=True,
        )
        all_results[f"BP (max_iter={max_iter})"] = r_bp
        save_results_csv(r_bp, os.path.join(RESULTS_DIR, f"exp3_bp_N{N}_R0.5.csv"))

        shannon_db = find_capacity_limit(rate)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={rate})",
            os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_bler.png"),
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={max_iter})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.png"), dpi=150)
        plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.pdf"))
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    from validate import main as run_validate

    run_validate()
    main()
