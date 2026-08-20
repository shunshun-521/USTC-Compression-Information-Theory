"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 最大迭代次数 50，min-sum 近似（alpha=0.9375），含早停
"""
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
from utils import save_results_csv, plot_bler_curves, find_capacity_limit


def run_unit_tests():
    """BP 译码器基本功能校验。"""
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    N = 8
    frozen_bits = np.zeros(N, dtype=int)
    u = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    bp = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, iters = bp.decode(llr)
    assert len(u_hat) == N
    assert 1 <= iters <= 50
    assert np.array_equal(u, u_hat)
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(5.0, 10.5, 0.25)

if __name__ == "__main__":
    run_unit_tests()

    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\n{'=' * 60}\n实验三 N={N}\n{'=' * 60}")
        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        scl_decoder_obj = SCLDecoder(N, frozen_bits, list_size=4)

        def scl_d(llr_ch, _scl=scl_decoder_obj):
            u, pm = _scl.decode(llr_ch)
            return u, None

        r_scl = run_simulation(
            N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
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
