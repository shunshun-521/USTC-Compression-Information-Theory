"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validate import run_validate

run_validate()

from construction import ga_construction
from decoder_sc import decode_sc_from_channel_llr
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs("results", exist_ok=True)

N_LIST = [int(x) for x in os.environ.get("POLAR_N_LIST", "256,512").split(",")]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(1.0, 5.5, float(os.environ.get("POLAR_EB_STEP", "0.25")))
if os.environ.get("POLAR_QUICK", "0") == "1":
    EB_N0_RANGE = np.array([2.0, 3.0, 4.0])
    MAX_FRAMES = min(MAX_FRAMES, 2000)
    MIN_ERRORS = min(MIN_ERRORS, 25)

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    def sc_d(llr_in):
        return decode_sc_from_channel_llr(llr_in, frozen_bits), None

    r_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results["SC"] = r_sc
    save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

    def scl_d(llr_in):
        return SCLDecoder(N, frozen_bits, list_size=4).decode(llr_in)

    r_scl = run_simulation(
        N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results["SCL (L=4)"] = r_scl
    save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_in):
        return bp_decoder.decode(llr_in)

    r_bp = run_simulation(
        N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
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
