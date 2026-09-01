"""从已有 CSV 生成 exp1 图表（若仿真已部分完成）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

os.makedirs("results", exist_ok=True)
all_results = {}
for N in [256, 512, 1024]:
    path = f"results/exp1_sc_N{N}_R0.5.csv"
    if os.path.exists(path):
        all_results[f"SC, N={N}, K={N//2}"] = load_results_csv(path)

if all_results:
    shannon_db = find_capacity_limit(0.5)
    plot_bler_curves(
        all_results,
        title="SC Decoder BLER vs Eb/N0 (R=0.5)",
        save_path="results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )
    print("fig1 已生成")
