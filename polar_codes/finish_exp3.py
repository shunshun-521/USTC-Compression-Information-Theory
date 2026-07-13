"""补全实验三缺失结果（快速参数）"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs("results", exist_ok=True)
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
EB_N0 = np.arange(1.0, 5.5, 0.25)
EB_N0_SCL = np.arange(1.5, 4.5, 0.5)

for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    sc_csv = f"results/exp3_sc_N{N}_R0.5.csv"
    if os.path.exists(sc_csv):
        all_results["SC"] = load_results_csv(sc_csv)
    else:
        r = run_simulation(
            N, K, EB_N0,
            lambda llr: (sc_decode(llr, frozen_bits), None),
            "sc", 30000, 100, info_indices=info_idx,
        )
        save_results_csv(r, sc_csv)
        all_results["SC"] = r

    scl_csv = f"results/exp3_scl_N{N}_R0.5.csv"
    if not os.path.exists(scl_csv):
        print(f"Running SCL N={N}")
        r = run_simulation(
            N, K, EB_N0_SCL,
            lambda llr: (
                SCLDecoder(N, frozen_bits.astype(bool), 4).decode(llr)[0], None
            ),
            "scl", 400, 20, info_indices=info_idx,
        )
        save_results_csv(r, scl_csv)
    all_results["SCL (L=4)"] = load_results_csv(scl_csv)

    bp_csv = f"results/exp3_bp_N{N}_R0.5.csv"
    if not os.path.exists(bp_csv):
        print(f"Running BP N={N}")
        bp = BPDecoder(N, frozen_bits.astype(bool), max_iter=MAX_ITER)
        r = run_simulation(
            N, K, EB_N0,
            lambda llr: bp.decode(llr),
            "bp", 20000, 100, info_indices=info_idx,
        )
        save_results_csv(r, bp_csv)
    r_bp = load_results_csv(bp_csv)
    all_results[f"BP (max_iter={MAX_ITER})"] = r_bp

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        f"results/fig3_bp_N{N}_bler.png",
        shannon_limit_db=shannon_db,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x["eb_n0_db"] for x in r_bp], [x["avg_iters"] for x in r_bp], "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

print("finish_exp3 done")
