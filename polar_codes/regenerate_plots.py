"""从已有 CSV 结果重新生成所有图表"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import load_results_csv, plot_bler_curves, find_capacity_limit


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    rate = 0.5
    shannon_db = find_capacity_limit(rate)

    # 实验一
    exp1 = {}
    for n in (256, 512, 1024):
        path = os.path.join(results_dir, f"exp1_sc_N{n}_R0.5.csv")
        if os.path.exists(path):
            exp1[f"SC, N={n}, K={n // 2}"] = load_results_csv(path)
    if exp1:
        plot_bler_curves(
            exp1,
            title=f"SC Decoder BLER vs Eb/N0 (R={rate})",
            save_path=os.path.join(results_dir, "fig1_sc_bler.png"),
            shannon_limit_db=shannon_db,
        )

    # 实验二
    exp2 = {}
    mapping = {
        "SC (L=1)": "exp2_sc_N512_R0.5.csv",
        "SCL (L=2)": "exp2_scl_L2_N512_R0.5.csv",
        "SCL (L=4)": "exp2_scl_L4_N512_R0.5.csv",
        "SCL (L=8)": "exp2_scl_L8_N512_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": "exp2_cascl_L8_N512_R0.5.csv",
    }
    for label, fname in mapping.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            exp2[label] = load_results_csv(path)
    if exp2:
        plot_bler_curves(
            exp2,
            title="SCL vs SC BLER (N=512, R=0.5)",
            save_path=os.path.join(results_dir, "fig2_scl_bler.png"),
            shannon_limit_db=shannon_db,
        )
        labels = list(exp2.keys())
        avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in exp2.values()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, avg_times)
        ax.set_xlabel("Decoder")
        ax.set_ylabel("Avg Decode Time (ms)")
        ax.set_title("Decoding Time vs List Size (N=512)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "fig2_decode_time.png"), dpi=150)
        plt.savefig(os.path.join(results_dir, "fig2_decode_time.pdf"))
        plt.close()

    # 实验三
    for n in (256, 512):
        exp3 = {}
        for dec, fname in [
            ("SC", f"exp3_sc_N{n}_R0.5.csv"),
            ("SCL (L=4)", f"exp3_scl_N{n}_R0.5.csv"),
            (f"BP (max_iter=50)", f"exp3_bp_N{n}_R0.5.csv"),
        ]:
            path = os.path.join(results_dir, fname)
            if os.path.exists(path):
                exp3[dec] = load_results_csv(path)
        if exp3:
            plot_bler_curves(
                exp3,
                title=f"SC vs SCL vs BP (N={n}, R={rate})",
                save_path=os.path.join(results_dir, f"fig3_bp_N{n}_bler.png"),
                shannon_limit_db=shannon_db,
            )
        bp_path = os.path.join(results_dir, f"exp3_bp_N{n}_R0.5.csv")
        if os.path.exists(bp_path):
            r_bp = load_results_csv(bp_path)
            eb = [r["eb_n0_db"] for r in r_bp]
            iters = [r["avg_iters"] for r in r_bp if r["avg_iters"] is not None]
            if iters:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eb[: len(iters)], iters, "o-", color="purple")
                ax.set_xlabel("Eb/N0 (dB)")
                ax.set_ylabel("Avg Iterations")
                ax.set_title(f"BP Average Iterations (N={n}, max_iter=50)")
                ax.grid(True, alpha=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"fig3_bp_N{n}_iters.png"), dpi=150)
                plt.savefig(os.path.join(results_dir, f"fig3_bp_N{n}_iters.pdf"))
                plt.close()

    print("图表已重新生成。")


if __name__ == "__main__":
    main()
