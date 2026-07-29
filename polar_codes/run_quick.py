"""
快速生成实验结果（缩减仿真量，用于验证流程）。
完整仿真请运行 run_exp1.py / run_exp2.py / run_exp3.py。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs('results', exist_ok=True)

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 3000
MIN_ERRORS = 30
EB_N0_RANGE = np.arange(1.0, 5.0, 0.5)

# frozen sets
save_frozen_set_info([256, 512, 1024], None, DESIGN_EBN0, 'results/frozen_sets.txt')

# exp1
all_exp1 = {}
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    results = run_simulation(
        N, K, EB_N0_RANGE,
        lambda l, fb=frozen_bits: (sc_decode(l, fb), None),
        'sc', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    label = f'SC, N={N}, K={K}'
    all_exp1[label] = results
    save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

shannon_db = find_capacity_limit(RATE)
plot_bler_curves(all_exp1, f'SC Decoder BLER vs Eb/N0 (R={RATE})',
                 'results/fig1_sc_bler.png', shannon_limit_db=shannon_db)

# exp2
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
all_exp2 = {}

results_sc = run_simulation(
    N, K, EB_N0_RANGE,
    lambda l: (sc_decode(l, frozen_bits), None),
    'sc', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
)
all_exp2['SC (L=1)'] = results_sc
save_results_csv(results_sc, f'results/exp2_sc_N{N}_R0.5.csv')

for L in [2, 4, 8]:
    results = run_simulation(
        N, K, EB_N0_RANGE,
        lambda l, _L=L: (SCLDecoder(N, frozen_bits, list_size=_L).decode(l)[0], None),
        'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_exp2[f'SCL (L={L})'] = results
    save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

results_cascl = run_simulation(
    N, K, EB_N0_RANGE,
    lambda l: (SCLDecoder(N, frozen_bits, list_size=8, crc_length=8).decode(l)[0], None),
    'scl', MAX_FRAMES, MIN_ERRORS, crc_length=8,
    info_indices=info_idx, verbose=True,
)
all_exp2['CA-SCL (L=8, CRC=8)'] = results_cascl
save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')

plot_bler_curves(all_exp2, f'SCL vs SC BLER (N={N}, R={RATE})',
                 'results/fig2_scl_bler.png', shannon_limit_db=shannon_db)

labels = list(all_exp2.keys())
avg_times = [np.mean([r['avg_decode_time'] for r in v]) * 1000 for v in all_exp2.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, avg_times)
ax.set_xlabel('Decoder')
ax.set_ylabel('Avg Decode Time (ms)')
ax.set_title(f'Decoding Time vs List Size (N={N})')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('results/fig2_decode_time.png', dpi=150)
plt.savefig('results/fig2_decode_time.pdf')
plt.close()

# exp3
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_exp3 = {}

    r_sc = run_simulation(N, K, EB_N0_RANGE, lambda l: (sc_decode(l, frozen_bits), None),
                          'sc', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True)
    all_exp3['SC'] = r_sc
    save_results_csv(r_sc, f'results/exp3_sc_N{N}_R0.5.csv')

    r_scl = run_simulation(
        N, K, EB_N0_RANGE,
        lambda l: (SCLDecoder(N, frozen_bits, list_size=4).decode(l)[0], None),
        'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_exp3['SCL (L=4)'] = r_scl
    save_results_csv(r_scl, f'results/exp3_scl_N{N}_R0.5.csv')

    bp = BPDecoder(N, frozen_bits, max_iter=50)
    r_bp = run_simulation(N, K, EB_N0_RANGE, lambda l: bp.decode(l),
                          'bp', 800, 15, info_indices=info_idx, verbose=True)
    all_exp3['BP (max_iter=50)'] = r_bp
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

    plot_bler_curves(all_exp3, f'SC vs SCL vs BP (N={N}, R={RATE})',
                     f'results/fig3_bp_N{N}_bler.png', shannon_limit_db=shannon_db)

    eb = [r['eb_n0_db'] for r in r_bp]
    iters = [r['avg_iters'] for r in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, iters, 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Average Iterations (N={N})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('快速仿真完成，结果已保存至 results/')
