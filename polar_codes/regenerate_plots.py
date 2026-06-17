"""从已有 CSV 结果重新生成所有图表（无需重跑仿真）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
shannon_db = find_capacity_limit(RATE)


def regen_fig1():
    all_results = {}
    for N in [256, 512, 1024]:
        path = os.path.join(RESULTS_DIR, f"exp1_sc_N{N}_R0.5.csv")
        if os.path.exists(path):
            all_results[f"SC, N={N}, K={N // 2}"] = load_results_csv(path)
    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path=os.path.join(RESULTS_DIR, "fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )
    print("fig1 done")


def regen_fig2():
    all_results = {}
    mapping = {
        "SC (L=1)": "exp2_sc_N512_R0.5.csv",
        "SCL (L=2)": "exp2_scl_L2_N512_R0.5.csv",
        "SCL (L=4)": "exp2_scl_L4_N512_R0.5.csv",
        "SCL (L=8)": "exp2_scl_L8_N512_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": "exp2_cascl_L8_N512_R0.5.csv",
    }
    for label, fname in mapping.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            all_results[label] = load_results_csv(path)

    plot_bler_curves(
        all_results,
        title=f"SCL vs SC BLER (N=512, R={RATE})",
        save_path=os.path.join(RESULTS_DIR, "fig2_scl_bler.png"),
        shannon_limit_db=shannon_db,
    )

    labels = list(all_results.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.png"), dpi=150)
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.pdf"))
    plt.close()
    print("fig2 done")


def regen_fig3():
    for N in [256, 512]:
        all_results = {}
        for dec, fname in [
            ("SC", f"exp3_sc_N{N}_R0.5.csv"),
            ("SCL (L=4)", f"exp3_scl_N{N}_R0.5.csv"),
            ("BP (max_iter=50)", f"exp3_bp_N{N}_R0.5.csv"),
        ]:
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                all_results[dec] = load_results_csv(path)

        plot_bler_curves(
            all_results,
            title=f"SC vs SCL vs BP (N={N}, R={RATE})",
            save_path=os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_bler.png"),
            shannon_limit_db=shannon_db,
        )

        bp_path = os.path.join(RESULTS_DIR, f"exp3_bp_N{N}_R0.5.csv")
        if os.path.exists(bp_path):
            r_bp = load_results_csv(bp_path)
            eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
            avg_iters = [r["avg_iters"] for r in r_bp]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("Avg Iterations")
            ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.png"), dpi=150)
            plt.savefig(os.path.join(RESULTS_DIR, f"fig3_bp_N{N}_iters.pdf"))
            plt.close()
    print("fig3 done")


if __name__ == "__main__":
    regen_fig1()
    regen_fig2()
    regen_fig3()
    print("All plots regenerated.")
