"""从已有 CSV 结果重新生成所有 BLER 曲线与柱状图。"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
shannon_db = find_capacity_limit(RATE)


def fig1():
    all_results = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "exp1_sc_N*_R0.5.csv"))):
        n = int(path.split("_N")[1].split("_")[0])
        all_results[f"SC, N={n}, K={n // 2}"] = load_results_csv(path)
    if all_results:
        plot_bler_curves(
            all_results,
            title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
            save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
            shannon_limit_db=shannon_db,
        )
        print("fig1_sc_bler.png/.pdf")


def fig2():
    mapping = {
        "SC (L=1)": "exp2_sc_N512_R0.5.csv",
        "SCL (L=2)": "exp2_scl_L2_N512_R0.5.csv",
        "SCL (L=4)": "exp2_scl_L4_N512_R0.5.csv",
        "SCL (L=8)": "exp2_scl_L8_N512_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": "exp2_cascl_L8_N512_R0.5.csv",
    }
    all_results = {}
    for label, fname in mapping.items():
        path = os.path.join(RESULTS, fname)
        if os.path.isfile(path):
            all_results[label] = load_results_csv(path)
    if all_results:
        plot_bler_curves(
            all_results,
            title=f"SCL vs SC BLER (N=512, R={RATE})",
            save_path=os.path.join(RESULTS, "fig2_scl_bler.png"),
            shannon_limit_db=shannon_db,
        )
        print("fig2_scl_bler.png/.pdf")

        labels = list(all_results.keys())
        avg_times = [
            np.mean([r["avg_decode_time"] for r in v]) * 1000
            for v in all_results.values()
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, avg_times)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Decoder")
        ax.set_ylabel("Avg Decode Time (ms)")
        ax.set_title("Decoding Time vs List Size (N=512)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "fig2_decode_time.png"), dpi=150)
        plt.savefig(os.path.join(RESULTS, "fig2_decode_time.pdf"))
        plt.close()
        print("fig2_decode_time.png/.pdf")


def fig3():
    for n in (256, 512):
        all_results = {}
        for dec, suffix in (
            ("SC", "sc"),
            ("SCL (L=4)", "scl"),
            (f"BP (max_iter=50)", "bp"),
        ):
            path = os.path.join(RESULTS, f"exp3_{suffix}_N{n}_R0.5.csv")
            if os.path.isfile(path):
                all_results[dec] = load_results_csv(path)
        if all_results:
            plot_bler_curves(
                all_results,
                title=f"SC vs SCL vs BP (N={n}, R={RATE})",
                save_path=os.path.join(RESULTS, f"fig3_bp_N{n}_bler.png"),
                shannon_limit_db=shannon_db,
            )
            print(f"fig3_bp_N{n}_bler.png/.pdf")

        bp_path = os.path.join(RESULTS, f"exp3_bp_N{n}_R0.5.csv")
        if os.path.isfile(bp_path):
            r_bp = load_results_csv(bp_path)
            eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
            avg_iters = [r["avg_iters"] for r in r_bp]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("Avg Iterations")
            ax.set_title(f"BP Average Iterations (N={n}, max_iter=50)")
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{n}_iters.png"), dpi=150)
            plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{n}_iters.pdf"))
            plt.close()
            print(f"fig3_bp_N{n}_iters.png/.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("全部图表已重新生成。")
