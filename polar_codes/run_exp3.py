"""
实验三：BP 译码
- 码长 N = 256, 512
- 与 SC、SCL（L=4）对比
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validate import run_all as run_validation

run_validation()

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation, get_simulation_limits
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs("results", exist_ok=True)

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES, MIN_ERRORS = get_simulation_limits(100000, 100)
EB_N0_RANGE = np.arange(4.0, 12.5, 1.0)

if os.environ.get("POLAR_FAST_SIM"):
    EB_N0_RANGE = np.arange(6.0, 14.0, 2.0)
    MAX_FRAMES = min(MAX_FRAMES, 1500)
    MIN_ERRORS = min(MIN_ERRORS, 10)
    N_LIST = [256]

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    all_results = {}

    def sc_d(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print(f"\n{'=' * 60}\nN={N}: SC 仿真\n{'=' * 60}")
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
        u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    print(f"\nN={N}: SCL 仿真")
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

    print(f"\nN={N}: BP 仿真")
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
    if N == 256:
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            "results/fig3_bp_bler.png",
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
