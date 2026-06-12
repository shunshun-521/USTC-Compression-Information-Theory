"""
实验三：BP 译码
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(2)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_sent)
    llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
    u_t = np.zeros(N, dtype=int)
    u_t[info_idx] = rng.integers(0, 2, K)
    llr_t = compute_llr(bpsk_modulate(polar_encode(u_t)), eb_n0_to_sigma(10.0, K / N))
    u_bp, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr_t)
    assert u_bp.shape == (N,) and iters >= 1
    assert np.all(u_bp[frozen_bits.astype(bool)] == 0), "BP 冻结位应为 0"
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    n_list = [256] if quick else [256, 512]
    rate = 0.5
    design_ebn0 = 2.5
    max_iter = 50
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "2000" if quick else "100000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "20" if quick else "100"))
    eb_n0_range = np.arange(6.0, 10.5, 1.0) if quick else np.arange(5.0, 11.5, 0.5)

    for N in n_list:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, design_ebn0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\n实验三 SC: N={N}")
        r_sc = run_simulation(
            N, K, eb_n0_range, sc_d, "sc", max_frames, min_errors,
            info_indices=info_idx, frozen_bits=frozen_bits,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        scl_obj = SCLDecoder(N, frozen_bits, list_size=4)

        def scl_d(llr_ch):
            u, _ = scl_obj.decode(llr_ch)
            return u, None

        print(f"实验三 SCL: N={N}")
        r_scl = run_simulation(
            N, K, eb_n0_range, scl_d, "scl", max_frames, min_errors,
            info_indices=info_idx, frozen_bits=frozen_bits,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=max_iter)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"实验三 BP: N={N}")
        r_bp = run_simulation(
            N, K, eb_n0_range, bp_d, "bp", max_frames, min_errors,
            info_indices=info_idx, frozen_bits=frozen_bits,
        )
        all_results[f"BP (max_iter={max_iter})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        shannon_db = find_capacity_limit(rate)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={rate})",
            f"results/fig3_bp_N{N}_bler.png",
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
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
