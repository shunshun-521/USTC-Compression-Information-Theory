#!/usr/bin/env python3
"""从已有 CSV 结果重新生成图表。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = "results"
RATE = 0.5


def main():
    shannon_db = find_capacity_limit(RATE)

    exp1 = {}
    for n in [256, 512, 1024]:
        path = f"{RESULTS}/exp1_sc_N{n}_R0.5.csv"
        if os.path.exists(path):
            exp1[f"SC, N={n}, K={n // 2}"] = load_results_csv(path)
    if exp1:
        plot_bler_curves(exp1, f"SC Decoder BLER vs Eb/N0 (R={RATE})",
                         f"{RESULTS}/fig1_sc_bler.png", shannon_limit_db=shannon_db)
        print("fig1 已生成")

    exp2 = {}
    mapping = {
        "SC (L=1)": f"{RESULTS}/exp2_sc_N512_R0.5.csv",
        "SCL (L=2)": f"{RESULTS}/exp2_scl_L2_N512_R0.5.csv",
        "SCL (L=4)": f"{RESULTS}/exp2_scl_L4_N512_R0.5.csv",
        "CA-SCL (L=4, CRC=8)": f"{RESULTS}/exp2_cascl_L4_N512_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": f"{RESULTS}/exp2_cascl_L8_N512_R0.5.csv",
    }
    for label, path in mapping.items():
        if os.path.exists(path):
            exp2[label] = load_results_csv(path)
    if exp2:
        plot_bler_curves(exp2, f"SCL vs SC BLER (N=512, R={RATE})",
                         f"{RESULTS}/fig2_scl_bler.png", shannon_limit_db=shannon_db)
        labels = list(exp2.keys())
        avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in exp2.values()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, avg_times)
        ax.set_xlabel("Decoder")
        ax.set_ylabel("Avg Decode Time (ms)")
        ax.set_title("Decoding Time vs List Size (N=512)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(f"{RESULTS}/fig2_decode_time.png", dpi=150)
        plt.savefig(f"{RESULTS}/fig2_decode_time.pdf")
        plt.close()
        print("fig2 已生成")


if __name__ == "__main__":
    main()
