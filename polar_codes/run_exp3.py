"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from decoder_sc import construct_frozen_ga
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit
from run_exp1 import run_unit_tests, _quick_mode

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_unit_tests()

    N_LIST = [256] if _quick_mode() else [256, 512]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_ITER = 50
    MAX_FRAMES = 3000 if _quick_mode() else 100000
    MIN_ERRORS = 15 if _quick_mode() else 100
    EB_N0_RANGE = (
        np.arange(1.5, 3.5, 0.5) if _quick_mode() else np.arange(1.0, 5.5, 0.25)
    )

    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}\n实验三: N={N}, K={K}\n{'=' * 60}")
        info_idx, _, _ = construct_frozen_ga(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, frozen_bits=frozen_bits,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        scl = SCLDecoder(N, frozen_bits, list_size=4)

        def scl_d(llr_ch):
            u, pm = scl.decode(llr_ch)
            return u, None

        r_scl = run_simulation(
            N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, frozen_bits=frozen_bits,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
            info_indices=info_idx, frozen_bits=frozen_bits,
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

    # 合并图 fig3_bp_bler（取 N=512 若存在，否则 N=256）
    try:
        from utils import load_results_csv

        merged = {}
        for N in ([512, 256] if not _quick_mode() else [256]):
            path = f"results/exp3_bp_N{N}_R0.5.csv"
            if os.path.exists(path):
                merged[f"BP, N={N}"] = load_results_csv(path)
        if merged:
            plot_bler_curves(
                merged,
                "BP Decoder BLER",
                "results/fig3_bp_bler.png",
                shannon_limit_db=find_capacity_limit(RATE),
            )
    except Exception:
        pass

    print("\n实验三完成。")
