"""快速重跑实验三 BP 部分（代表性 SNR 点）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 15
MAX_FRAMES = 5000
MIN_ERRORS = 20
EB_N0_RANGE = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.25])

for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {
        'SC': load_results_csv(f'results/exp3_sc_N{N}_R0.5.csv'),
        'SCL (L=4)': load_results_csv(f'results/exp3_scl_N{N}_R0.5.csv'),
    }

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        return bp_decoder.decode(llr_ch)

    print(f"\n快速 BP 仿真: N={N}")
    r_bp = run_simulation(
        N, K, EB_N0_RANGE, bp_d, 'bp', MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results[f'BP (max_iter={MAX_ITER})'] = r_bp
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f'SC vs SCL vs BP (N={N}, R={RATE})',
        f'results/fig3_bp_N{N}_bler.png',
        shannon_limit_db=shannon_db,
    )

    eb = [r['eb_n0_db'] for r in r_bp]
    it = [r['avg_iters'] for r in r_bp]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb, it, 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Average Iterations (N={N}, max_iter={MAX_ITER})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('快速 BP 仿真完成。')
