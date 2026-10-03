"""
从 results/*.csv 重新生成 BLER 曲线与译码时间图（无需重跑蒙特卡洛）。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
shannon_db = find_capacity_limit(RATE)


def _load(path):
    p = os.path.join(RESULTS, path)
    if os.path.isfile(p):
        return load_results_csv(p)
    return None


def fig_exp1():
    curves = {}
    for N in (256, 512, 1024):
        data = _load(f"exp1_sc_N{N}_R0.5.csv")
        if data:
            curves[f"SC, N={N}, K={N // 2}"] = data
    if curves:
        plot_bler_curves(
            curves,
            f"SC Decoder BLER vs Eb/N0 (R={RATE})",
            os.path.join(RESULTS, "fig1_sc_bler.png"),
            shannon_limit_db=shannon_db,
        )


def fig_exp2():
    curves = {}
    mapping = [
        ("exp2_sc_N512_R0.5.csv", "SC (L=1)"),
        ("exp2_scl_L2_N512_R0.5.csv", "SCL (L=2)"),
        ("exp2_scl_L4_N512_R0.5.csv", "SCL (L=4)"),
        ("exp2_scl_N512_R0.5.csv", "SCL (L=4)"),
        ("exp2_scl_L8_N512_R0.5.csv", "SCL (L=8)"),
        ("exp2_cascl_L8_N512_R0.5.csv", "CA-SCL (L=8, CRC=8)"),
    ]
    seen = set()
    for fname, label in mapping:
        if label in seen:
            continue
        data = _load(fname)
        if data:
            curves[label] = data
            seen.add(label)
    if not curves:
        return
    plot_bler_curves(
        curves,
        f"SCL vs SC BLER (N=512, R={RATE})",
        os.path.join(RESULTS, "fig2_scl_bler.png"),
        shannon_limit_db=shannon_db,
    )
    labels = list(curves.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in curves[k]]) * 1000 for k in labels
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


def fig_exp3():
    for N in (256, 512):
        curves = {}
        for key, label in (
            (f"exp3_sc_N{N}_R0.5.csv", "SC"),
            (f"exp3_scl_N{N}_R0.5.csv", "SCL (L=4)"),
            (f"exp3_bp_N{N}_R0.5.csv", f"BP (max_iter=50)"),
        ):
            data = _load(key)
            if data:
                curves[label] = data
        if not curves:
            continue
        plot_bler_curves(
            curves,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            os.path.join(RESULTS, f"fig3_bp_N{N}_bler.png"),
            shannon_limit_db=shannon_db,
        )
        bp = _load(f"exp3_bp_N{N}_R0.5.csv")
        if bp and bp[0].get("avg_iters") is not None:
            eb = [r["eb_n0_db"] for r in bp]
            iters = [r["avg_iters"] for r in bp]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb, iters, "o-", color="purple")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("Avg Iterations")
            ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{N}_iters.png"), dpi=150)
            plt.savefig(os.path.join(RESULTS, f"fig3_bp_N{N}_iters.pdf"))
            plt.close()


if __name__ == "__main__":
    fig_exp1()
    fig_exp2()
    fig_exp3()
    print("图表已写入 results/")
