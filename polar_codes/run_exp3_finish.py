"""完成实验三剩余仿真。"""
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
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv
from verify import run_unit_tests

os.makedirs('results', exist_ok=True)
run_unit_tests()

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
SNR_SC = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
SNR_FAST = np.array([1.0, 2.0, 3.0])

for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sc_path = f'results/exp3_sc_N{N}_R0.5.csv'
    if os.path.exists(sc_path):
        r_sc = load_results_csv(sc_path)
    else:
        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None
        print(f'\n实验三 SC: N={N}')
        r_sc = run_simulation(
            N, K, SNR_SC, sc_d, 'sc', 10000, 30,
            info_indices=info_idx, verbose=True,
        )
        save_results_csv(r_sc, sc_path)

    def scl_d(llr_ch):
        u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    print(f'\n实验三 SCL: N={N}')
    r_scl = run_simulation(
        N, K, SNR_FAST, scl_d, 'scl', 300 if N > 256 else 800, 15,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(r_scl, f'results/exp3_scl_N{N}_R0.5.csv')

    bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

    def bp_d(llr_ch):
        u_hat, num_iters = bp_decoder.decode(llr_ch)
        return u_hat, num_iters

    print(f'\n实验三 BP: N={N}')
    r_bp = run_simulation(
        N, K, SNR_SC, bp_d, 'bp', 5000, 20,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

    all_results = {
        'SC': r_sc,
        'SCL (L=4)': r_scl,
        f'BP (max_iter={MAX_ITER})': r_bp,
    }
    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f'SC vs SCL vs BP (N={N}, R={RATE})',
        f'results/fig3_bp_N{N}_bler.png',
        shannon_limit_db=shannon_db,
    )

    eb_n0_vals = [r['eb_n0_db'] for r in r_bp]
    avg_iters = [r['avg_iters'] for r in r_bp]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eb_n0_vals, avg_iters, 'o-', color='purple')
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel('Avg Iterations')
    ax.set_title(f'BP Average Iterations (N={N}, max_iter={MAX_ITER})')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
    plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
    plt.close()

print('\n实验三完成。')
