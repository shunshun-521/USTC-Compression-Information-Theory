"""从 results/*.csv 重新生成 BLER 曲线图（无需重跑蒙特卡洛仿真）。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RATE = 0.5
shannon_db = find_capacity_limit(RATE)


def _group_csv(pattern):
    out = {}
    for name in sorted(os.listdir(RESULTS)):
        m = re.match(pattern, name)
        if not m:
            continue
        label = m.group(1)
        out[label] = load_results_csv(os.path.join(RESULTS, name))
    return out


def main():
    exp1 = _group_csv(r"exp1_sc_N(\d+)_R0\.5\.csv")
    if exp1:
        plot_bler_curves(
            {f"SC, N={n}, K={int(n)//2}": exp1[n] for n in sorted(exp1, key=int)},
            title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
            save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
            shannon_limit_db=shannon_db,
        )

    exp2_labels = {
        "exp2_sc_N512_R0.5.csv": "SC (L=1)",
        "exp2_scl_L2_N512_R0.5.csv": "SCL (L=2)",
        "exp2_scl_L4_N512_R0.5.csv": "SCL (L=4)",
        "exp2_scl_L8_N512_R0.5.csv": "SCL (L=8)",
        "exp2_cascl_L8_N512_R0.5.csv": "CA-SCL (L=8, CRC=8)",
    }
    exp2 = {}
    for fname, label in exp2_labels.items():
        path = os.path.join(RESULTS, fname)
        if os.path.isfile(path):
            exp2[label] = load_results_csv(path)
    if exp2:
        plot_bler_curves(
            exp2,
            title="SCL vs SC BLER (N=512, R=0.5)",
            save_path=os.path.join(RESULTS, "fig2_scl_bler.png"),
            shannon_limit_db=shannon_db,
        )

    for N in (256, 512):
        exp3 = {}
        mapping = {
            f"exp3_sc_N{N}_R0.5.csv": "SC",
            f"exp3_scl_N{N}_R0.5.csv": "SCL (L=4)",
            f"exp3_bp_N{N}_R0.5.csv": "BP (max_iter=50)",
        }
        for fname, label in mapping.items():
            path = os.path.join(RESULTS, fname)
            if os.path.isfile(path):
                exp3[label] = load_results_csv(path)
        if exp3:
            plot_bler_curves(
                exp3,
                title=f"SC vs SCL vs BP (N={N}, R={RATE})",
                save_path=os.path.join(RESULTS, f"fig3_bp_N{N}_bler.png"),
                shannon_limit_db=shannon_db,
            )

    print("Figures written under", RESULTS)


if __name__ == "__main__":
    main()
