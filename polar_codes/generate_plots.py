"""从已有 CSV 结果重新生成所有图表。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_capacity_limit, load_results_csv, plot_bler_curves

os.makedirs('results', exist_ok=True)
RATE = 0.5
shannon_db = find_capacity_limit(RATE)

# 实验一
all_results = {}
for N in [256, 512, 1024]:
    path = f'results/exp1_sc_N{N}_R0.5.csv'
    if os.path.exists(path):
        all_results[f'SC, N={N}, K={N // 2}'] = load_results_csv(path)
if all_results:
    plot_bler_curves(
        all_results,
        title=f'SC Decoder BLER vs Eb/N0 (R={RATE})',
        save_path='results/fig1_sc_bler.png',
        shannon_limit_db=shannon_db,
    )
    print('Generated fig1_sc_bler')

# 实验二
all_results = {}
mapping = {
    'results/exp2_sc_N512_R0.5.csv': 'SC (L=1)',
    'results/exp2_scl_L2_N512_R0.5.csv': 'SCL (L=2)',
    'results/exp2_scl_L4_N512_R0.5.csv': 'SCL (L=4)',
    'results/exp2_scl_L8_N512_R0.5.csv': 'SCL (L=8)',
    'results/exp2_cascl_L8_N512_R0.5.csv': 'CA-SCL (L=8, CRC=8)',
}
for path, label in mapping.items():
    if os.path.exists(path):
        all_results[label] = load_results_csv(path)
if all_results:
    plot_bler_curves(
        all_results,
        f'SCL vs SC BLER (N=512, R={RATE})',
        'results/fig2_scl_bler.png',
        shannon_limit_db=shannon_db,
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
    print('Generated fig2')

# 实验三
for N in [256, 512]:
    all_results = {}
    for dec, fname in [('SC', 'sc'), ('SCL (L=4)', 'scl'), ('BP (max_iter=50)', 'bp')]:
        path = f'results/exp3_{fname}_N{N}_R0.5.csv'
        if os.path.exists(path):
            all_results[dec] = load_results_csv(path)
    if all_results:
        plot_bler_curves(
            all_results,
            f'SC vs SCL vs BP (N={N}, R={RATE})',
            f'results/fig3_bp_N{N}_bler.png',
            shannon_limit_db=shannon_db,
        )
        print(f'Generated fig3_bp_N{N}_bler')
    bp_path = f'results/exp3_bp_N{N}_R0.5.csv'
    if os.path.exists(bp_path):
        r_bp = load_results_csv(bp_path)
        eb_n0_vals = [r['eb_n0_db'] for r in r_bp]
        avg_iters = [r['avg_iters'] for r in r_bp if r['avg_iters'] is not None]
        if avg_iters:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(eb_n0_vals[:len(avg_iters)], avg_iters, 'o-', color='purple')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Avg Iterations')
            ax.set_title(f'BP Average Iterations (N={N}, max_iter=50)')
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
            plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
            plt.close()
            print(f'Generated fig3_bp_N{N}_iters')

print('Plot generation complete.')
