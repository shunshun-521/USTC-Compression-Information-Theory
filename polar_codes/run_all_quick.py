"""
快速运行全部实验（用于生成 results/ 输出，参数略缩减以控制运行时间）
完整参数请直接运行 run_exp1/2/3.py
"""
import os
import sys

os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs('results', exist_ok=True)

MAX_FRAMES = 8000
MIN_ERRORS = 50
DESIGN_EBN0 = 2.5
RATE = 0.5

# ========== 实验一 ==========
N_LIST = [256, 512, 1024]
EB1 = np.arange(0.0, 5.5, 0.5)
save_frozen_set_info(N_LIST, None, DESIGN_EBN0, 'results/frozen_sets.txt')
all1 = {}
for N in N_LIST:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    fb = np.ones(N, dtype=bool); fb[info_idx] = False
    print(f'Exp1 N={N}')
    r = run_simulation(N, K, EB1, lambda l: (sc_decode(l, fb), None), 'sc',
                       MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all1[f'SC, N={N}, K={K}'] = r
    save_results_csv(r, f'results/exp1_sc_N{N}_R0.5.csv')
plot_bler_curves(all1, f'SC Decoder BLER (R={RATE})', 'results/fig1_sc_bler.png',
                 find_capacity_limit(RATE))

# ========== 实验二 ==========
N = 512; K = N // 2; CRC_LEN = 8
EB2 = np.arange(1.0, 5.5, 0.5)
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
fb = np.ones(N, dtype=bool); fb[info_idx] = False
all2 = {}
r = run_simulation(N, K, EB2, lambda l: (sc_decode(l, fb), None), 'sc',
                   MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
all2['SC (L=1)'] = r
save_results_csv(r, 'results/exp2_sc_N512_R0.5.csv')
for L in [2, 4, 8]:
    print(f'Exp2 SCL L={L}')
    r = run_simulation(N, K, EB2,
                       lambda l, _L=L: (SCLDecoder(N, fb, _L).decode(l)[0], None),
                       'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all2[f'SCL (L={L})'] = r
    save_results_csv(r, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')
print('Exp2 CA-SCL')
r = run_simulation(N, K, EB2,
                   lambda l: (SCLDecoder(N, fb, 8, CRC_LEN, info_idx).decode(l)[0], None),
                   'scl', MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LEN, info_indices=info_idx)
all2[f'CA-SCL (L=8, CRC={CRC_LEN})'] = r
save_results_csv(r, 'results/exp2_cascl_L8_N512_R0.5.csv')
plot_bler_curves(all2, f'SCL vs SC (N={N})', 'results/fig2_scl_bler.png', find_capacity_limit(RATE))
labels = list(all2.keys())
times = [np.mean([x['avg_decode_time'] for x in v]) * 1000 for v in all2.values()]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, times); ax.set_ylabel('Avg Decode Time (ms)'); ax.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig('results/fig2_decode_time.png', dpi=150)
plt.savefig('results/fig2_decode_time.pdf'); plt.close()

# ========== 实验三 ==========
EB3 = np.arange(1.0, 5.5, 0.5)
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    fb = np.ones(N, dtype=bool); fb[info_idx] = False
    all3 = {}
    print(f'Exp3 N={N} SC')
    r = run_simulation(N, K, EB3, lambda l: (sc_decode(l, fb), None), 'sc',
                       MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all3['SC'] = r; save_results_csv(r, f'results/exp3_sc_N{N}_R0.5.csv')
    print(f'Exp3 N={N} SCL')
    r = run_simulation(N, K, EB3,
                       lambda l: (SCLDecoder(N, fb, 4).decode(l)[0], None),
                       'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all3['SCL (L=4)'] = r; save_results_csv(r, f'results/exp3_scl_N{N}_R0.5.csv')
    bp = BPDecoder(N, fb, max_iter=50)
    print(f'Exp3 N={N} BP')
    r = run_simulation(N, K, EB3, lambda l: bp.decode(l), 'bp',
                       MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all3['BP (max_iter=50)'] = r; save_results_csv(r, f'results/exp3_bp_N{N}_R0.5.csv')
    plot_bler_curves(all3, f'SC vs SCL vs BP (N={N})', f'results/fig3_bp_N{N}_bler.png',
                     find_capacity_limit(RATE))
    eb = [x['eb_n0_db'] for x in r]; it = [x['avg_iters'] for x in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, it, 'o-', color='purple'); ax.set_xlabel('Eb/N0 (dB)'); ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Avg Iterations (N={N})'); ax.grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf'); plt.close()

print('全部实验完成。')
