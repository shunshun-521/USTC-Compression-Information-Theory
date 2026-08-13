"""
快速仿真（用于验证流程，参数较完整仿真更小）
完整仿真请运行 run_exp1.py / run_exp2.py / run_exp3.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit
from validate import run_unit_tests

os.makedirs('results', exist_ok=True)
run_unit_tests()

QUICK_MAX_FRAMES = 5000
QUICK_MIN_ERRORS = 30
EB_N0 = np.arange(1.0, 4.5, 0.5)
RATE = 0.5
DESIGN_EBN0 = 2.5

save_frozen_set_info([256, 512], None, DESIGN_EBN0, 'results/frozen_sets.txt')

# Exp1 quick
all_results = {}
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N, K, EB_N0, decoder, 'sc',
        QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results[f'SC, N={N}'] = results
    save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

plot_bler_curves(all_results, 'SC BLER (quick)', 'results/fig1_sc_bler.png',
                 find_capacity_limit(RATE))

# Exp2 quick
N = 512
K = N // 2
info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
all_results = {}

def sc_decoder(llr_ch):
    return sc_decode(llr_ch, frozen_bits), None

all_results['SC (L=1)'] = run_simulation(
    N, K, EB_N0, sc_decoder, 'sc', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
    info_indices=info_idx, verbose=True,
)
save_results_csv(all_results['SC (L=1)'], f'results/exp2_sc_N{N}_R0.5.csv')

for L in [2, 4, 8]:
    def scl_decoder(llr_ch, _L=L):
        u, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
        return u, None
    label = f'SCL (L={L})'
    all_results[label] = run_simulation(
        N, K, EB_N0, scl_decoder, 'scl', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(all_results[label], f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

plot_bler_curves(all_results, 'SCL BLER (quick)', 'results/fig2_scl_bler.png',
                 find_capacity_limit(RATE))

# Exp3 quick
for N in [256, 512]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    def sc_d(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    r_sc = run_simulation(N, K, EB_N0, sc_d, 'sc', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                          info_indices=info_idx, verbose=True)
    all_results['SC'] = r_sc
    save_results_csv(r_sc, f'results/exp3_sc_N{N}_R0.5.csv')

    def scl_d(llr_ch):
        u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
        return u, None

    r_scl = run_simulation(N, K, EB_N0, scl_d, 'scl', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                           info_indices=info_idx, verbose=True)
    all_results['SCL (L=4)'] = r_scl
    save_results_csv(r_scl, f'results/exp3_scl_N{N}_R0.5.csv')

    bp = BPDecoder(N, frozen_bits, max_iter=50)

    def bp_d(llr_ch):
        return bp.decode(llr_ch)

    r_bp = run_simulation(N, K, EB_N0, bp_d, 'bp', QUICK_MAX_FRAMES, QUICK_MIN_ERRORS,
                          info_indices=info_idx, verbose=True)
    all_results['BP'] = r_bp
    save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

    plot_bler_curves(all_results, f'Exp3 N={N} (quick)', f'results/fig3_bp_N{N}_bler.png',
                     find_capacity_limit(RATE))

print('快速仿真完成。')
