"""从已有 CSV 结果生成缺失图表（无需重跑仿真）。"""
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

os.makedirs("results", exist_ok=True)
RATE = 0.5
shannon_db = find_capacity_limit(RATE)
print(f"Shannon limit (R={RATE}): {shannon_db:.3f} dB")

# 实验一：SC
exp1 = {}
for path in sorted(glob.glob("results/exp1_sc_N*_R0.5.csv")):
    m = re.search(r"N(\d+)", path)
    if m:
        N = m.group(1)
        exp1[f"SC, N={N}, K={int(N)//2}"] = load_results_csv(path)
if exp1:
    plot_bler_curves(
        exp1,
        f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        "results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )
    print("fig1_sc_bler.png/.pdf")

# 实验二：SCL
exp2_labels = [
    ("results/exp2_sc_N512_R0.5.csv", "SC (L=1)"),
    ("results/exp2_scl_L2_N512_R0.5.csv", "SCL (L=2)"),
    ("results/exp2_scl_L4_N512_R0.5.csv", "SCL (L=4)"),
    ("results/exp2_scl_L8_N512_R0.5.csv", "SCL (L=8)"),
    ("results/exp2_cascl_L8_N512_R0.5.csv", "CA-SCL (L=8, CRC=8)"),
]
exp2 = {}
for path, label in exp2_labels:
    if os.path.isfile(path):
        exp2[label] = load_results_csv(path)
if exp2:
    plot_bler_curves(
        exp2,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
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
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()
    print("fig2_scl_bler.png/.pdf, fig2_decode_time.png/.pdf")

# 实验三：按码长
for N in (256, 512):
    paths = {
        "SC": f"results/exp3_sc_N{N}_R0.5.csv",
        "SCL (L=4)": f"results/exp3_scl_N{N}_R0.5.csv",
        f"BP (max_iter=50)": f"results/exp3_bp_N{N}_R0.5.csv",
    }
    exp3 = {k: load_results_csv(v) for k, v in paths.items() if os.path.isfile(v)}
    if not exp3:
        continue
    plot_bler_curves(
        exp3,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        f"results/fig3_bp_N{N}_bler.png",
        shannon_limit_db=shannon_db,
    )
    bp_path = f"results/exp3_bp_N{N}_R0.5.csv"
    if os.path.isfile(bp_path):
        r_bp = load_results_csv(bp_path)
        eb = [r["eb_n0_db"] for r in r_bp]
        iters = [r["avg_iters"] for r in r_bp if r.get("avg_iters")]
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
    print(f"fig3_bp_N{N}_*.png/.pdf")

print("图表生成完成。")
