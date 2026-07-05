"""重新绘制实验一 BLER 曲线（含 N=1024）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

RATE = 0.5
all_results = {}
for N in [256, 512, 1024]:
    path = f'results/exp1_sc_N{N}_R0.5.csv'
    if os.path.exists(path):
        all_results[f'SC, N={N}, K={N//2}'] = load_results_csv(path)

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(
    all_results,
    title=f'SC Decoder BLER vs Eb/N0 (R={RATE})',
    save_path='results/fig1_sc_bler.png',
    shannon_limit_db=shannon_db,
)
print('fig1 已更新')
