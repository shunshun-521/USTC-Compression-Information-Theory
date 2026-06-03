"""从已有 CSV 重新生成 BLER 曲线图"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RATE = 0.5
shannon = find_capacity_limit(RATE)

# 实验一
r1 = {
    f"SC, N={N}, K={N//2}": load_results_csv(f"results/exp1_sc_N{N}_R0.5.csv")
    for N in [256, 512, 1024]
}
plot_bler_curves(r1, f"SC Decoder BLER vs Eb/N0 (R={RATE})", "results/fig1_sc_bler.png", shannon)

# 实验二
labels2 = ["SC (L=1)", "SCL (L=2)", "SCL (L=4)", "SCL (L=8)", "CA-SCL (L=8, CRC=8)"]
files2 = [
    "results/exp2_sc_N512_R0.5.csv",
    "results/exp2_scl_L2_N512_R0.5.csv",
    "results/exp2_scl_L4_N512_R0.5.csv",
    "results/exp2_scl_L8_N512_R0.5.csv",
    "results/exp2_cascl_L8_N512_R0.5.csv",
]
r2 = {lb: load_results_csv(fp) for lb, fp in zip(labels2, files2)}
plot_bler_curves(r2, "SCL vs SC BLER (N=512, R=0.5)", "results/fig2_scl_bler.png", shannon)

labels = list(r2.keys())
avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in r2.values()]
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

# 实验三
for N in [256, 512]:
    r3 = {
        "SC": load_results_csv(f"results/exp3_sc_N{N}_R0.5.csv"),
        "SCL (L=4)": load_results_csv(f"results/exp3_scl_N{N}_R0.5.csv"),
        "BP (max_iter=50)": load_results_csv(f"results/exp3_bp_N{N}_R0.5.csv"),
    }
    plot_bler_curves(
        r3,
        f"SC vs SCL vs BP (N={N}, R={RATE})",
        f"results/fig3_bp_N{N}_bler.png",
        shannon,
    )
    r_bp = load_results_csv(f"results/exp3_bp_N{N}_R0.5.csv")
    eb = [r["eb_n0_db"] for r in r_bp]
    iters = [r["avg_iters"] for r in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, iters, "o-", color="purple")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Avg Iterations")
    ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
    plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
    plt.close()

print("绘图完成。")
