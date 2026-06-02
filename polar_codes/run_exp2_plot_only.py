"""用已有 CSV 绘制实验二曲线（无需重跑仿真）"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_results_csv, plot_bler_curves, find_capacity_limit

all_results = {}
for label, path in [
    ('SC (L=1)', 'results/exp2_sc_N512_R0.5.csv'),
    ('SCL (L=2)', 'results/exp2_scl_L2_N512_R0.5.csv'),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

if all_results:
    plot_bler_curves(all_results, 'SCL vs SC BLER (N=512, R=0.5)',
                     'results/fig2_scl_bler.png', find_capacity_limit(0.5))
    labels = list(all_results.keys())
    avg_times = [np.mean([r['avg_decode_time'] for r in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel('Decoder'); ax.set_ylabel('Avg Decode Time (ms)')
    ax.set_title('Decoding Time (N=512)')
    plt.tight_layout()
    plt.savefig('results/fig2_decode_time.png', dpi=150)
    plt.savefig('results/fig2_decode_time.pdf')
    plt.close()
    print('fig2 已生成（SC + SCL L=2）')
