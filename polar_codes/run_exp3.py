"""
实验三：BP 译码
- 码长 N = 256, 512
- 码率 R = 1/2
- 最大迭代次数 50，min-sum 近似（alpha=0.9375），含早停
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validate import run_all as run_validation
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

run_validation()

N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_MIN = float(os.environ.get("POLAR_EB_MIN", "1.0"))
EB_MAX = float(os.environ.get("POLAR_EB_MAX", "5.25"))
EB_STEP = float(os.environ.get("POLAR_EB_STEP", "0.25"))
EB_N0_RANGE = np.arange(EB_MIN, EB_MAX + 1e-9, EB_STEP)

if os.environ.get("POLAR_FAST"):
    MAX_FRAMES = min(MAX_FRAMES, 3000)
    MIN_ERRORS = min(MIN_ERRORS, 15)

for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    all_results = {}

    def sc_d(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print(f"\n{'=' * 60}\nExp3 SC N={N}\n{'=' * 60}")
    r_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results["SC"] = r_sc
    save_results_csv(r_sc, os.path.join(RESULTS_DIR, f"exp3_sc_N{N}_R0.5.csv"))

    def scl_d(llr_ch):
        u, pm = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    print(f"\nExp3 SCL N={N}")
    r_scl = run_simulation(
        N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results["SCL (L=4)"] = r_scl
    save_results_csv(r_scl, os.path.join(RESULTS_DIR, f"exp3_scl_N{N}_R0.5.csv"))

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        u_hat, num_iters = bp_decoder.decode(llr_ch)
        return u_hat, num_iters

    print(f"\nExp3 BP N={N}")
    r_bp = run_simulation(
        N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
    save_results_csv(r_bp, os.path.join(RESULTS_DIR, f"exp3_bp_N{N}_R0.5.csv"))

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_bler.png"),
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
    plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.png"), dpi=150)
    plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.pdf"))
    plt.close()

print("\n实验三完成。")
