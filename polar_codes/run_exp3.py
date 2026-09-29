"""
实验三：BP 译码
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validate import run_all as run_validate
from construction import info_set_for_codec
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), "results")

if not os.environ.get("POLAR_SKIP_VALIDATE"):
    run_validate()

N_LIST = [int(x) for x in os.environ.get("POLAR_EXP3_N_LIST", "256,512").split(",")]
RATE = 0.5
DESIGN_EBN0 = float(os.environ.get("POLAR_DESIGN_EBN0", "2.5"))
MAX_ITER = int(os.environ.get("POLAR_BP_MAX_ITER", "50"))
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

for N in N_LIST:
    K = N // 2
    info_idx = info_set_for_codec(N, K)
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
        frozen_bits=frozen_bits,
        info_indices=info_idx,
    )
    all_results["SC"] = r_sc
    save_results_csv(r_sc, os.path.join(RESULTS, f"exp3_sc_N{N}_R0.5.csv"))

    def scl_d(llr_ch):
        u, pm = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    r_scl = run_simulation(
        N,
        K,
        EB_N0_RANGE,
        scl_d,
        "scl",
        MAX_FRAMES,
        MIN_ERRORS,
        frozen_bits=frozen_bits,
        info_indices=info_idx,
    )
    all_results["SCL (L=4)"] = r_scl
    save_results_csv(r_scl, os.path.join(RESULTS, f"exp3_scl_N{N}_R0.5.csv"))

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
        frozen_bits=frozen_bits,
        info_indices=info_idx,
    )
    all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
    save_results_csv(r_bp, os.path.join(RESULTS, f"exp3_bp_N{N}_R0.5.csv"))

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        os.path.join(RESULTS, f"fig3_bp_N{N}_bler.png"),
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
    plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{N}_iters.png"), dpi=150)
    plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{N}_iters.pdf"))
    plt.close()

print("\n实验三完成。")
