"""生成全部结果文件（缩减规模，用于 CI/自动化验证）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from verify import run_all
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

run_all()
os.makedirs('results', exist_ok=True)

RATE = 0.5
DESIGN = 2.5
EB = [3.0, 4.0]
MF, ME = 20, 3

# exp1
save_frozen_set_info([256, 512, 1024], None, DESIGN, 'results/frozen_sets.txt')
all1 = {}
for N in [256, 512, 1024]:
    K = N // 2
    mf = 25 if N <= 512 else 10
    info, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int); fb[info] = 0
    r = run_simulation(N, K, EB, lambda l: (sc_decode(l, fb), None), 'sc', mf, ME, info_indices=info, verbose=True)
    all1[f'SC, N={N}, K={K}'] = r
    save_results_csv(r, f'results/exp1_sc_N{N}_R0.5.csv')
plot_bler_curves(all1, f'SC BLER (R={RATE})', 'results/fig1_sc_bler.png', find_capacity_limit(RATE))

# exp2
N = 512; K = N // 2
info, _, _ = ga_construction(N, K, DESIGN)
fb = np.ones(N, dtype=int); fb[info] = 0
all2 = {}
r_sc = run_simulation(N, K, EB, lambda l: (sc_decode(l, fb), None), 'sc', MF, ME, info_indices=info, verbose=True)
all2['SC (L=1)'] = r_sc
save_results_csv(r_sc, f'results/exp2_sc_N{N}_R0.5.csv')
for L in [2, 4]:
    mf2 = 15 if L > 2 else 20
    r = run_simulation(N, K, EB, lambda l, _L=L: (SCLDecoder(N, fb, _L).decode(l)[0], None), 'scl', mf2, ME, info_indices=info, verbose=True)
    all2[f'SCL (L={L})'] = r
    save_results_csv(r, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')
    if L == 4:
        save_results_csv(r, f'results/exp2_scl_N{N}_R0.5.csv')
r_c = run_simulation(N, K, EB, lambda l: (SCLDecoder(N, fb, 4, 8).decode(l)[0], None), 'scl', MF, ME, crc_length=8, info_indices=info, verbose=True)
all2['CA-SCL (L=4, CRC=8)'] = r_c
save_results_csv(r_c, f'results/exp2_cascl_L4_N{N}_R0.5.csv')
plot_bler_curves(all2, f'SCL vs SC (N={N})', 'results/fig2_scl_bler.png', find_capacity_limit(RATE))
if plt:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(all2.keys()), [np.mean([x['avg_decode_time'] for x in v]) * 1000 for v in all2.values()])
    ax.set_title('Decode Time'); plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig('results/fig2_decode_time.png', dpi=150); plt.savefig('results/fig2_decode_time.pdf'); plt.close()

# exp3
for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int); fb[info] = 0
    all3 = {}
    all3['SC'] = run_simulation(N, K, EB, lambda l: (sc_decode(l, fb), None), 'sc', MF, ME, info_indices=info, verbose=True)
    save_results_csv(all3['SC'], f'results/exp3_sc_N{N}_R0.5.csv')
    all3['SCL (L=4)'] = run_simulation(N, K, EB, lambda l: (SCLDecoder(N, fb, 4).decode(l)[0], None), 'scl', MF, ME, info_indices=info, verbose=True)
    save_results_csv(all3['SCL (L=4)'], f'results/exp3_scl_N{N}_R0.5.csv')
    bp = BPDecoder(N, fb)
    r_bp = run_simulation(N, K, EB, lambda l: bp.decode(l), 'bp', MF, ME, info_indices=info, verbose=True)
    all3['BP (max_iter=50)'] = r_bp
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')
    plot_bler_curves(all3, f'SC/SCL/BP N={N}', f'results/fig3_bp_N{N}_bler.png', find_capacity_limit(RATE))
    if plt:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x['eb_n0_db'] for x in r_bp], [x['avg_iters'] for x in r_bp], 'o-')
        ax.set_xlabel('Eb/N0 (dB)'); ax.set_ylabel('Avg Iterations'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150); plt.savefig(f'results/fig3_bp_N{N}_iters.pdf'); plt.close()

print('全部结果已生成。')
