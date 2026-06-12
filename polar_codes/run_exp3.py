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
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, 0.5)
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors == 0, f"SC 测试失败: {errors}/100"
    print("单元测试通过。")


def _sim_params():
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    return {
        "max_frames": 300 if quick else 100000,
        "min_errors": 10 if quick else 100,
        "eb_n0_range": np.arange(2.0, 4.5, 0.5) if quick else np.arange(1.0, 5.5, 0.25),
        "n_list": [256] if quick else [256, 512],
        "max_iter": 50,
    }


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    params = _sim_params()
    RATE = 0.5
    DESIGN_EBN0 = 2.5

    for N in params["n_list"]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"\n实验三 N={N}: SC")
        r_sc = run_simulation(
            N,
            K,
            params["eb_n0_range"],
            sc_d,
            "sc",
            params["max_frames"],
            params["min_errors"],
            info_indices=info_idx,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr_ch):
            u, pm = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        print(f"\n实验三 N={N}: SCL L=4")
        r_scl = run_simulation(
            N,
            K,
            params["eb_n0_range"],
            scl_d,
            "scl",
            params["max_frames"],
            params["min_errors"],
            info_indices=info_idx,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=params["max_iter"])

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"\n实验三 N={N}: BP")
        r_bp = run_simulation(
            N,
            K,
            params["eb_n0_range"],
            bp_d,
            "bp",
            params["max_frames"],
            params["min_errors"],
            info_indices=info_idx,
        )
        all_results[f"BP (max_iter={params['max_iter']})"] = r_bp
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
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={params['max_iter']})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("\n实验三完成。")


if __name__ == "__main__":
    main()
