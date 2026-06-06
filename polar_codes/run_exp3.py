"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 与 SC、SCL（L=4）对比
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decode,
)
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), (u @ build_generator_matrix(4)) % 2)

    n, k = 64, 32
    info_idx, _, _ = ga_construction(n, k, 2.5)
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(10.0, k / n)

    for _ in range(100):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        llr = reorder_llr_for_decode(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng), sigma),
            n,
        )
        assert np.array_equal(sc_decode(llr, frozen_bits), u_test)

    scl = SCLDecoder(n, frozen_bits, list_size=1)
    for _ in range(20):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        llr = reorder_llr_for_decode(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng), sigma),
            n,
        )
        assert np.array_equal(sc_decode(llr, frozen_bits), scl.decode(llr)[0])

    bp = BPDecoder(n, frozen_bits, max_iter=50)
    ok = 0
    for _ in range(20):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        y = awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng)
        uh, _ = bp.decode(compute_llr(y, sigma))
        ok += int(np.array_equal(uh, u_test))
    assert ok >= 18, f"BP 译码通过率过低: {ok}/20"
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

    for n_val in n_list:
        k_val = n_val // 2
        info_idx, _, _ = ga_construction(n_val, k_val, design_ebn0)
        frozen_bits = np.ones(n_val, dtype=bool)
        frozen_bits[info_idx] = False

        all_results = {}

        def sc_d(llr_ch, _fb=frozen_bits):
            return sc_decode(llr_ch, _fb), None

        r_sc = run_simulation(
            n_val,
            k_val,
            eb_n0_range,
            sc_d,
            "sc",
            max_frames,
            min_errors,
            info_indices=info_idx,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{n_val}_R0.5.csv")

        def scl_d(llr_ch, _n=n_val, _fb=frozen_bits):
            u_hat, _ = SCLDecoder(_n, _fb, list_size=4).decode(llr_ch)
            return u_hat, None

        r_scl = run_simulation(
            n_val,
            k_val,
            eb_n0_range,
            scl_d,
            "scl",
            max_frames,
            min_errors,
            info_indices=info_idx,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{n_val}_R0.5.csv")

        bp_decoder = BPDecoder(n_val, frozen_bits, max_iter=max_iter)

        def bp_d(llr_ch, _bp=bp_decoder):
            u_hat, num_iters = _bp.decode(llr_ch)
            return u_hat, num_iters

        r_bp = run_simulation(
            n_val,
            k_val,
            eb_n0_range,
            bp_d,
            "bp",
            max_frames,
            min_errors,
            info_indices=info_idx,
        )
        all_results[f"BP (max_iter={max_iter})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{n_val}_R0.5.csv")

        shannon_db = find_capacity_limit(rate)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={n_val}, R={rate})",
            f"results/fig3_bp_N{n_val}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={n_val}, max_iter={max_iter})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{n_val}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{n_val}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
