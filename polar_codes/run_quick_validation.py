"""
快速验证脚本：生成 results/ 目录下的全部输出文件（参数适度降低以缩短运行时间）。
完整参数仿真请直接运行 run_exp1.py / run_exp2.py / run_exp3.py。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs('results', exist_ok=True)

QUICK_MAX_FRAMES = 3000
QUICK_MIN_ERRORS = 30
DESIGN_EBN0 = 2.5
RATE = 0.5

# frozen sets
save_frozen_set_info([256, 512, 1024], None, DESIGN_EBN0, 'results/frozen_sets.txt')

# exp1
all_sc = {}
for N in [256, 512, 1024]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, fb=frozen_bits):
        return sc_decode(llr_ch, fb), None

    results = run_simulation(
        N, K, np.arange(0.0, 5.5, 0.5), decoder, 'sc',
        QUICK_MAX_FRAMES, QUICK_MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_sc[f'SC, N={N}, K={K}'] = results
    save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

plot_bler_curves(all_sc, 'SC Decoder BLER vs Eb/N0 (R=0.5)',
                 'results/fig1_sc_bler.png', find_capacity_limit(RATE))

# exp2
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
eb = np.arange(1.0, 5.5, 0.5)
all_scl = {}

def sc_d(llr):
    return sc_decode(llr, frozen_bits), None

r = run_simulation(N, K, eb, sc_d, 'sc', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                   info_indices=info_idx, verbose=True)
all_scl['SC (L=1)'] = r
save_results_csv(r, f'results/exp2_sc_N{N}_R0.5.csv')

for L in [2, 4, 8]:
    def scl_d(llr, _L=L):
        u, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr)
        return u, None
    r = run_simulation(N, K, eb, scl_d, 'scl', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                       info_indices=info_idx, verbose=True)
    all_scl[f'SCL (L={L})'] = r
    save_results_csv(r, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

def cascl_d(llr):
    u, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=8).decode(llr)
    return u, None

r = run_simulation(N, K, eb, cascl_d, 'scl', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                   crc_length=8, info_indices=info_idx, verbose=True)
all_scl['CA-SCL (L=8, CRC=8)'] = r
save_results_csv(r, f'results/exp2_cascl_L8_N{N}_R0.5.csv')
save_results_csv(r, f'results/exp2_scl_N{N}_R0.5.csv')

plot_bler_curves(all_scl, f'SCL vs SC BLER (N={N})', 'results/fig2_scl_bler.png',
                 find_capacity_limit(RATE))
labels = list(all_scl.keys())
avg_times = [np.mean([x['avg_decode_time'] for x in v]) * 1000 for v in all_scl.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, avg_times)
ax.set_xlabel('Decoder')
ax.set_ylabel('Avg Decode Time (ms)')
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
    all_bp = {}

    def sc_d2(llr):
        return sc_decode(llr, frozen_bits), None

    r_sc = run_simulation(N, K, eb, sc_d2, 'sc', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                          info_indices=info_idx, verbose=True)
    all_bp['SC'] = r_sc
    save_results_csv(r_sc, f'results/exp3_sc_N{N}_R0.5.csv')

    def scl_d2(llr):
        u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
        return u, None

    r_scl = run_simulation(N, K, eb, scl_d2, 'scl', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                           info_indices=info_idx, verbose=True)
    all_bp['SCL (L=4)'] = r_scl
    save_results_csv(r_scl, f'results/exp3_scl_N{N}_R0.5.csv')

    bp = BPDecoder(N, frozen_bits, max_iter=50)

    def bp_d(llr):
        u, it = bp.decode(llr)
        return u, it

    r_bp = run_simulation(N, K, eb, bp_d, 'bp', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                          info_indices=info_idx, verbose=True)
    all_bp['BP (max_iter=50)'] = r_bp
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

    plot_bler_curves(all_bp, f'SC vs SCL vs BP (N={N})', f'results/fig3_bp_N{N}_bler.png',
                     find_capacity_limit(RATE))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x['eb_n0_db'] for x in r_bp], [x['avg_iters'] for x in r_bp], 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('快速验证完成，结果已写入 results/')
