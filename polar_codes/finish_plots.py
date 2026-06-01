"""从已有 CSV 重新生成 BLER 曲线图"""
import glob
import os
import re

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RATE = 0.5


def main():
    os.makedirs("results", exist_ok=True)
    exp1 = {}
    for path in sorted(glob.glob("results/exp1_sc_N*_R0.5.csv")):
        m = re.search(r"N(\d+)", path)
        if m:
            n = int(m.group(1))
            exp1[f"SC, N={n}, K={n // 2}"] = load_results_csv(path)
    if exp1:
        plot_bler_curves(
            exp1,
            f"SC Decoder BLER vs Eb/N0 (R={RATE})",
            "results/fig1_sc_bler.png",
            find_capacity_limit(RATE),
        )

    exp2 = {}
    for path in sorted(glob.glob("results/exp2_*.csv")):
        label = os.path.basename(path).replace(".csv", "").replace("exp2_", "").replace("_R0.5", "")
        exp2[label] = load_results_csv(path)
    if exp2:
        n_tag = "512" if any("N512" in k for k in exp2) else "128"
        plot_bler_curves(
            exp2,
            f"SCL vs SC BLER (N={n_tag}, R={RATE})",
            "results/fig2_scl_bler.png",
            find_capacity_limit(RATE),
        )

    for n in [256, 512]:
        exp3 = {}
        for kind in ["sc", "scl", "bp"]:
            p = f"results/exp3_{kind}_N{n}_R0.5.csv"
            if os.path.isfile(p):
                exp3[kind.upper() if kind == "sc" else f"{kind.upper()} ({'L=4' if kind=='scl' else 'max_iter=50'})"] = load_results_csv(p)
        if exp3:
            plot_bler_curves(
                exp3,
                f"SC vs SCL vs BP (N={n}, R={RATE})",
                f"results/fig3_bp_N{n}_bler.png",
                find_capacity_limit(RATE),
            )


if __name__ == "__main__":
    main()
