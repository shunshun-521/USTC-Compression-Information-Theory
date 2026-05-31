"""根据已有 CSV 重新绘制 BLER 曲线（更新香农限竖线）"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_results_csv, plot_bler_curves, find_capacity_limit

RATE = 0.5
shannon_db = find_capacity_limit(RATE)
print(f"Shannon limit Eb/N0 = {shannon_db:.3f} dB")

# 实验一
exp1 = {}
for N in [256, 512, 1024]:
    path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.path.exists(path):
        exp1[f"SC, N={N}, K={N//2}"] = load_results_csv(path)
if exp1:
    plot_bler_curves(
        exp1,
        f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        "results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )

# 实验二
exp2 = {}
mapping = {
    "SC (L=1)": "results/exp2_sc_N512_R0.5.csv",
    "SCL (L=2)": "results/exp2_scl_L2_N512_R0.5.csv",
    "SCL (L=4)": "results/exp2_scl_L4_N512_R0.5.csv",
    "SCL (L=8)": "results/exp2_scl_L8_N512_R0.5.csv",
    "CA-SCL (L=8, CRC=8)": "results/exp2_cascl_L8_N512_R0.5.csv",
}
for label, path in mapping.items():
    if os.path.exists(path):
        exp2[label] = load_results_csv(path)
if exp2:
    plot_bler_curves(
        exp2,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )

# 实验三
for N in [256, 512]:
    exp3 = {}
    for name, fn in [
        ("SC", f"exp3_sc_N{N}_R0.5.csv"),
        ("SCL (L=4)", f"exp3_scl_N{N}_R0.5.csv"),
        ("BP (max_iter=50)", f"exp3_bp_N{N}_R0.5.csv"),
    ]:
        path = f"results/{fn}"
        if os.path.exists(path):
            exp3[name] = load_results_csv(path)
    if exp3:
        plot_bler_curves(
            exp3,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

print("绘图完成。")
