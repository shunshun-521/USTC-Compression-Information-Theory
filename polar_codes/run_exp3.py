"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

os.makedirs("results", exist_ok=True)


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_sent)
    llr = 50 * bpsk_modulate(x)
    u_hat, _ = BPDecoder(N, frozen).decode(llr)
    assert np.array_equal(u_hat, u_sent), "BP 无损译码失败"
    print("单元测试通过。")


def _quick_env():
    if os.environ.get("POLAR_QUICK", "0") == "1":
        return 500, 20, np.arange(1.5, 4.0, 0.5)
    return 100000, 100, np.arange(1.0, 5.5, 0.25)


run_unit_tests()

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES, MIN_ERRORS, EB_N0_RANGE = _quick_env()

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}

    def sc_d(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    r_sc = run_simulation(
        N,
        K,
        EB_N0_RANGE,
        sc_d,
        "sc",
        MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info_idx,
    )
    all_results["SC"] = r_sc
    save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

    def scl_d(llr_ch):
        u, _pm = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    r_scl = run_simulation(
        N,
        K,
        EB_N0_RANGE,
        scl_d,
        "scl",
        MAX_FRAMES,
        MIN_ERRORS,
        info_indices=info_idx,
    )
    all_results["SCL (L=4)"] = r_scl
    save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        u_hat, num_iters = bp_decoder.decode(llr_ch)
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
