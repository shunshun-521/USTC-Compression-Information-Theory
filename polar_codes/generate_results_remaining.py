"""补全尚未生成的实验结果文件。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

os.makedirs('results', exist_ok=True)
EB = [3.0, 4.0]
RATE = 0.5

# exp2 remainder
N = 512
K = N // 2
info, _, _ = ga_construction(N, K, 2.5)
fb = np.ones(N, dtype=int)
fb[info] = 0
all2 = {}
if os.path.exists('results/exp2_sc_N512_R0.5.csv'):
    from utils import load_results_csv
    all2['SC (L=1)'] = load_results_csv('results/exp2_sc_N512_R0.5.csv')
else:
    r_sc = run_simulation(N, K, EB, lambda l: (sc_decode(l, fb), None), 'sc', 20, 3, info_indices=info, verbose=True)
    all2['SC (L=1)'] = r_sc
    save_results_csv(r_sc, 'results/exp2_sc_N512_R0.5.csv')

for L in [2, 4]:
    path = f'results/exp2_scl_L{L}_N512_R0.5.csv'
    if not os.path.exists(path):
        r = run_simulation(N, K, EB, lambda l, _L=L: (SCLDecoder(N, fb, _L).decode(l)[0], None), 'scl', 15, 3, info_indices=info, verbose=True)
        save_results_csv(r, path)
        if L == 4:
            save_results_csv(r, 'results/exp2_scl_N512_R0.5.csv')
    all2[f'SCL (L={L})'] = __import__('utils').load_results_csv(path if L != 4 else 'results/exp2_scl_N512_R0.5.csv')

path = 'results/exp2_cascl_L4_N512_R0.5.csv'
if not os.path.exists(path):
    r_c = run_simulation(N, K, EB, lambda l: (SCLDecoder(N, fb, 4, 8).decode(l)[0], None), 'scl', 15, 3, crc_length=8, info_indices=info, verbose=True)
    save_results_csv(r_c, path)
all2['CA-SCL (L=4, CRC=8)'] = __import__('utils').load_results_csv(path)
plot_bler_curves(all2, 'SCL vs SC', 'results/fig2_scl_bler.png', find_capacity_limit(RATE))
if plt:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(all2.keys()), [np.mean([x['avg_decode_time'] for x in v]) * 1000 for v in all2.values()])
    ax.set_title('Decode Time'); plt.xticks(rotation=25); plt.tight_layout()
    plt.savefig('results/fig2_decode_time.png', dpi=150); plt.savefig('results/fig2_decode_time.pdf'); plt.close()

# exp3
for N in [256, 512]:
    K = N // 2
    info, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info] = 0
    all3 = {}
    mf = 20 if N == 256 else 15
    for name, fn, dt in [
        ('SC', lambda l: (sc_decode(l, fb), None), 'sc'),
        ('SCL (L=4)', lambda l: (SCLDecoder(N, fb, 4).decode(l)[0], None), 'scl'),
    ]:
        p = f'results/exp3_{name.split()[0].lower()}_N{N}_R0.5.csv'.replace('(l=4)', 'scl')
        if name == 'SCL (L=4)':
            p = f'results/exp3_scl_N{N}_R0.5.csv'
        if not os.path.exists(p):
            r = run_simulation(N, K, EB, fn, dt, mf, 3, info_indices=info, verbose=True)
            save_results_csv(r, p)
        all3[name] = __import__('utils').load_results_csv(p)

    p_bp = f'results/exp3_bp_N{N}_R0.5.csv'
    if not os.path.exists(p_bp):
        bp = BPDecoder(N, fb)
        r_bp = run_simulation(N, K, EB, lambda l: bp.decode(l), 'bp', mf, 3, info_indices=info, verbose=True)
        save_results_csv(r_bp, p_bp)
    all3['BP (max_iter=50)'] = __import__('utils').load_results_csv(p_bp)
    plot_bler_curves(all3, f'SC/SCL/BP N={N}', f'results/fig3_bp_N{N}_bler.png', find_capacity_limit(RATE))
    if plt:
        r_bp = all3['BP (max_iter=50)']
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x['eb_n0_db'] for x in r_bp], [x['avg_iters'] for x in r_bp], 'o-')
        ax.set_xlabel('Eb/N0 (dB)'); ax.set_ylabel('Avg Iterations'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150); plt.savefig(f'results/fig3_bp_N{N}_iters.pdf'); plt.close()

print('补全完成')
