"""从 results/*.csv 重新生成所有 BLER 曲线与译码时间图（无需重跑仿真）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
N_EXP1 = [256, 512, 1024]


def _path(name):
    return os.path.join(RESULTS, name)


def fig_exp1():
    all_results = {}
    for N in N_EXP1:
        fp = _path(f"exp1_sc_N{N}_R0.5.csv")
        if os.path.isfile(fp):
            all_results[f"SC, N={N}, K={N // 2}"] = load_results_csv(fp)
    if not all_results:
        return
    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path=_path("fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )


def fig_exp2():
    all_results = {}
    mapping = [
        ("exp2_sc_N512_R0.5.csv", "SC (L=1)"),
        ("exp2_scl_L2_N512_R0.5.csv", "SCL (L=2)"),
        ("exp2_scl_L4_N512_R0.5.csv", "SCL (L=4)"),
        ("exp2_scl_L8_N512_R0.5.csv", "SCL (L=8)"),
        ("exp2_cascl_L8_N512_R0.5.csv", "CA-SCL (L=8, CRC=8)"),
    ]
    for fname, label in mapping:
        fp = _path(fname)
        if os.path.isfile(fp):
            all_results[label] = load_results_csv(fp)
    if not all_results:
        return
    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        title="SCL vs SC BLER (N=512, R=0.5)",
        save_path=_path("fig2_scl_bler.png"),
        shannon_limit_db=shannon_db,
    )
    labels = list(all_results.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in all_results[k]]) * 1000 for k in labels
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(_path("fig2_decode_time.png"), dpi=150)
    plt.savefig(_path("fig2_decode_time.pdf"))
    plt.close()


def fig_exp3():
    for N in (256, 512):
        all_results = {}
        for dec, label in [
            ("sc", "SC"),
            ("scl", "SCL (L=4)"),
            ("bp", f"BP (max_iter=50)"),
        ]:
            fp = _path(f"exp3_{dec}_N{N}_R0.5.csv")
            if os.path.isfile(fp):
                all_results[label] = load_results_csv(fp)
        if not all_results:
            continue
        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results,
            title=f"SC vs SCL vs BP (N={N}, R={RATE})",
            save_path=_path(f"fig3_bp_N{N}_bler.png"),
            shannon_limit_db=shannon_db,
        )
        bp_fp = _path(f"exp3_bp_N{N}_R0.5.csv")
        if os.path.isfile(bp_fp):
            r_bp = load_results_csv(bp_fp)
            eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
            avg_iters = [r["avg_iters"] for r in r_bp if r["avg_iters"] is not None]
            if avg_iters and len(avg_iters) == len(eb_n0_vals):
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
                ax.set_xlabel("Eb/N0 (dB)")
                ax.set_ylabel("Avg Iterations")
                ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
                ax.grid(True, alpha=0.4)
                plt.tight_layout()
                plt.savefig(_path(f"fig3_bp_N{N}_iters.png"), dpi=150)
                plt.savefig(_path(f"fig3_bp_N{N}_iters.pdf"))
                plt.close()


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    fig_exp1()
    fig_exp2()
    fig_exp3()
    print("图表已写入 results/")
