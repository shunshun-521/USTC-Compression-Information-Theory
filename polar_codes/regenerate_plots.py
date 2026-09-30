"""从 results/*.csv 重新生成 BLER 曲线图（无需重跑蒙特卡洛）。"""
import glob
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
shannon_db = find_capacity_limit(RATE)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    exp1 = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "exp1_sc_N*_R0.5.csv"))):
        n = path.split("exp1_sc_N")[1].split("_")[0]
        exp1[f"SC, N={n}, K={int(n)//2}"] = load_results_csv(path)
    if exp1:
        plot_bler_curves(
            exp1,
            title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
            save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
            shannon_limit_db=shannon_db,
        )
        print("Wrote fig1_sc_bler")

    exp2_files = {
        "SC (L=1)": "exp2_sc_N512_R0.5.csv",
        "SCL (L=2)": "exp2_scl_L2_N512_R0.5.csv",
        "SCL (L=4)": "exp2_scl_L4_N512_R0.5.csv",
        "SCL (L=8)": "exp2_scl_L8_N512_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": "exp2_cascl_L8_N512_R0.5.csv",
    }
    exp2 = {}
    for label, fname in exp2_files.items():
        path = os.path.join(RESULTS, fname)
        if os.path.isfile(path):
            exp2[label] = load_results_csv(path)
    src_l4 = os.path.join(RESULTS, "exp2_scl_L4_N512_R0.5.csv")
    dst = os.path.join(RESULTS, "exp2_scl_N512_R0.5.csv")
    if os.path.isfile(src_l4) and not os.path.isfile(dst):
        shutil.copy2(src_l4, dst)
    if exp2:
        plot_bler_curves(
            exp2,
            title=f"SCL vs SC BLER (N=512, R={RATE})",
            save_path=os.path.join(RESULTS, "fig2_scl_bler.png"),
            shannon_limit_db=shannon_db,
        )
        labels = list(exp2.keys())
        avg_times = [
            np.mean([r["avg_decode_time"] for r in exp2[k]]) * 1000 for k in labels
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, avg_times)
        ax.set_xlabel("Decoder")
        ax.set_ylabel("Avg Decode Time (ms)")
        ax.set_title("Decoding Time vs List Size (N=512)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "fig2_decode_time.png"), dpi=150)
        plt.savefig(os.path.join(RESULTS, "fig2_decode_time.pdf"))
        plt.close()
        print("Wrote fig2_scl_bler and fig2_decode_time")

    for n in ("256", "512"):
        exp3 = {}
        mapping = {
            "SC": f"exp3_sc_N{n}_R0.5.csv",
            "SCL (L=4)": f"exp3_scl_N{n}_R0.5.csv",
            f"BP (max_iter=50)": f"exp3_bp_N{n}_R0.5.csv",
        }
        for label, fname in mapping.items():
            path = os.path.join(RESULTS, fname)
            if os.path.isfile(path):
                exp3[label] = load_results_csv(path)
        if not exp3:
            continue
        plot_bler_curves(
            exp3,
            title=f"SC vs SCL vs BP (N={n}, R={RATE})",
            save_path=os.path.join(RESULTS, f"fig3_bp_N{n}_bler.png"),
            shannon_limit_db=shannon_db,
        )
        bp_path = os.path.join(RESULTS, f"exp3_bp_N{n}_R0.5.csv")
        if os.path.isfile(bp_path):
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
                plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{n}_iters.png"), dpi=150)
                plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{n}_iters.pdf"))
                plt.close()
        if n == "256" and exp3:
            plot_bler_curves(
                exp3,
                title=f"SC vs SCL vs BP (N={n}, R={RATE})",
                save_path=os.path.join(RESULTS, "fig3_bp_bler.png"),
                shannon_limit_db=shannon_db,
            )
        print(f"Wrote fig3 for N={n}")


if __name__ == "__main__":
    main()
