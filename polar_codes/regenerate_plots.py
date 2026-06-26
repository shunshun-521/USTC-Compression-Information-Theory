"""从已有 CSV 结果重新生成所有图表"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_results_csv, plot_bler_curves, find_capacity_limit

os.makedirs("results", exist_ok=True)
RATE = 0.5

# 实验一
exp1 = {}
for N in [256, 512, 1024]:
    path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.path.exists(path):
        exp1[f"SC, N={N}, K={N // 2}"] = load_results_csv(path)

if exp1:
    plot_bler_curves(
        exp1,
        f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        "results/fig1_sc_bler.png",
        shannon_limit_db=find_capacity_limit(RATE),
    )
    print("fig1_sc_bler.png/.pdf")

# 实验二
exp2_labels = {
    "results/exp2_sc_N512_R0.5.csv": "SC (L=1)",
    "results/exp2_scl_L2_N512_R0.5.csv": "SCL (L=2)",
    "results/exp2_scl_L4_N512_R0.5.csv": "SCL (L=4)",
    "results/exp2_scl_L8_N512_R0.5.csv": "SCL (L=8)",
    "results/exp2_cascl_L8_N512_R0.5.csv": "CA-SCL (L=8, CRC=8)",
}
exp2 = {}
for path, label in exp2_labels.items():
    if os.path.exists(path):
        exp2[label] = load_results_csv(path)

if exp2:
    plot_bler_curves(
        exp2,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
        shannon_limit_db=find_capacity_limit(RATE),
    )
    labels = list(exp2.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in exp2.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()
    print("fig2_scl_bler.png/.pdf, fig2_decode_time.png/.pdf")

# 实验三
for N in [256, 512]:
    exp3 = {}
    for key, fname in [
        ("SC", f"exp3_sc_N{N}_R0.5.csv"),
        ("SCL (L=4)", f"exp3_scl_N{N}_R0.5.csv"),
        (f"BP (max_iter=50)", f"exp3_bp_N{N}_R0.5.csv"),
    ]:
        path = f"results/{fname}"
        if os.path.exists(path):
            exp3[key] = load_results_csv(path)

    if exp3:
        plot_bler_curves(
            exp3,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=find_capacity_limit(RATE),
        )
        print(f"fig3_bp_N{N}_bler.png/.pdf")

    bp_path = f"results/exp3_bp_N{N}_R0.5.csv"
    if os.path.exists(bp_path):
        r_bp = load_results_csv(bp_path)
        eb = [r["eb_n0_db"] for r in r_bp]
        iters = [r["avg_iters"] for r in r_bp if r["avg_iters"] is not None]
        if iters:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb[: len(iters)], iters, "o-", color="purple")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("Avg Iterations")
            ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
            plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
            plt.close()
            print(f"fig3_bp_N{N}_iters.png/.pdf")

print("All plots regenerated.")
