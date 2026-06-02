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
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    n, k = 64, 32
    info, _, _ = ga_construction(n, k, 2.5)
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[info] = False
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    sigma = eb_n0_to_sigma(10.0, k / n)
    rng = np.random.default_rng(2)
    for _ in range(50):
        u_t = np.zeros(n, dtype=int)
        u_t[info] = rng.integers(0, 2, k)
        llr = compute_llr(bpsk_modulate(polar_encode(u_t)), sigma)
        uh, _ = BPDecoder(n, frozen_bits, max_iter=50).decode(llr)
        assert np.array_equal(uh, u_t)
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n_list = [256, 512]
    rate = 0.5
    design_ebn0 = 2.5
    max_iter = 50
    max_frames = 100000
    min_errors = 100
    eb_n0_range = np.arange(1.0, 5.5, 0.25)

    for n in n_list:
        k = n // 2
        info_idx, _, _ = ga_construction(n, k, design_ebn0)
        frozen_bits = np.ones(n, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\nN={n} SC")
        r_sc = run_simulation(
            n, k, eb_n0_range, sc_d, "sc", max_frames, min_errors, verbose=True
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{n}_R0.5.csv")

        def scl_d(llr_ch):
            u, _ = SCLDecoder(n, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        print(f"\nN={n} SCL L=4")
        r_scl = run_simulation(
            n, k, eb_n0_range, scl_d, "scl", max_frames, min_errors, verbose=True
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{n}_R0.5.csv")

        bp_decoder = BPDecoder(n, frozen_bits, max_iter=max_iter)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"\nN={n} BP")
        r_bp = run_simulation(
            n, k, eb_n0_range, bp_d, "bp", max_frames, min_errors, verbose=True
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
