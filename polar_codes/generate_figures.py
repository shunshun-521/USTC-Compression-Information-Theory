"""从已有 CSV 生成最终图表"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_results_csv, plot_bler_curves, find_capacity_limit
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('results', exist_ok=True)
RATE = 0.5
shannon_db = find_capacity_limit(RATE)

# 实验二图
all_results = {}
for label, path in [
    ('SC (L=1)', 'results/exp2_sc_N512_R0.5.csv'),
    ('SCL (L=2)', 'results/exp2_scl_L2_N512_R0.5.csv'),
    ('SCL (L=4)', 'results/exp2_scl_L4_N512_R0.5.csv'),
    ('SCL (L=8)', 'results/exp2_scl_L8_N512_R0.5.csv'),
    ('CA-SCL (L=8, CRC=8)', 'results/exp2_cascl_L8_N512_R0.5.csv'),
]:
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)

if all_results:
    plot_bler_curves(
        all_results, 'SCL vs SC BLER (N=512, R=0.5)',
        'results/fig2_scl_bler.png', shannon_limit_db=shannon_db,
    )
    labels = list(all_results.keys())
    avg_times = [np.mean([r['avg_decode_time'] for r in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel('Decoder')
    ax.set_ylabel('Avg Decode Time (ms)')
    ax.set_title('Decoding Time vs List Size (N=512)')
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig('results/fig2_decode_time.png', dpi=150)
    plt.savefig('results/fig2_decode_time.pdf')
    plt.close()

# 实验三 N=256 图（已有数据）
for N in [256, 512]:
    all_results3 = {}
    for label, suffix in [('SC', 'sc'), ('SCL (L=4)', 'scl'), ('BP', 'bp')]:
        path = f'results/exp3_{suffix}_N{N}_R0.5.csv'
        if os.path.exists(path):
            key = label if suffix != 'bp' else f'BP (max_iter=30)'
            all_results3[key] = load_results_csv(path)
    if len(all_results3) >= 2:
        plot_bler_curves(
            all_results3, f'SC vs SCL vs BP (N={N}, R=0.5)',
            f'results/fig3_bp_N{N}_bler.png', shannon_limit_db=shannon_db,
        )
    bp_path = f'results/exp3_bp_N{N}_R0.5.csv'
    if os.path.exists(bp_path):
        r_bp = load_results_csv(bp_path)
        eb = [r['eb_n0_db'] for r in r_bp]
        iters = [r['avg_iters'] for r in r_bp if r['avg_iters']]
        if iters:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb[:len(iters)], iters, 'o-', color='purple')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Avg Iterations')
            ax.set_title(f'BP Average Iterations (N={N})')
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
            plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
            plt.close()

print('图表已生成。')
