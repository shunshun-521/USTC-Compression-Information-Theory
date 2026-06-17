#!/usr/bin/env python3
"""从已有 CSV 结果重新生成 BLER 曲线图，无需重新仿真。"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    rate = 0.5
    shannon_db = find_capacity_limit(rate)

    exp1 = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "exp1_sc_N*_R0.5.csv"))):
        m = re.search(r"N(\d+)", path)
        if m:
            label = f"SC, N={m.group(1)}, K={int(m.group(1)) // 2}"
            exp1[label] = load_results_csv(path)
    if exp1:
        plot_bler_curves(exp1, f"SC Decoder BLER vs Eb/N0 (R={rate})",
                         os.path.join(results_dir, "fig1_sc_bler.png"), shannon_db)

    exp2 = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "exp2_*.csv"))):
        name = os.path.basename(path).replace(".csv", "")
        if "sc_" in name:
            label = "SC (L=1)"
        elif "cascl" in name:
            label = "CA-SCL (L=8, CRC=8)"
        elif "scl_L" in name:
            m = re.search(r"L(\d+)", name)
            label = f"SCL (L={m.group(1)})" if m else name
        else:
            label = name
        exp2[label] = load_results_csv(path)
    if exp2:
        plot_bler_curves(exp2, "SCL vs SC BLER", os.path.join(results_dir, "fig2_scl_bler.png"), shannon_db)

    for n in [256, 512]:
        exp3 = {}
        for kind, tag in [("sc", "SC"), ("scl", "SCL (L=4)"), ("bp", "BP (max_iter=50)")]:
            path = os.path.join(results_dir, f"exp3_{kind}_N{n}_R0.5.csv")
            if os.path.isfile(path):
                exp3[tag] = load_results_csv(path)
        if exp3:
            plot_bler_curves(exp3, f"SC vs SCL vs BP (N={n}, R={rate})",
                             os.path.join(results_dir, f"fig3_bp_N{n}_bler.png"), shannon_db)

    print("Plots regenerated in results/")


if __name__ == "__main__":
    main()
