"""
快速仿真脚本（用于验证，参数较原实验缩减）
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
from utils import (
    save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit,
)
from tests import run_unit_tests
import matplotlib.pyplot as plt


def main():
    os.makedirs('results', exist_ok=True)
    run_unit_tests()

    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = 3000
    MIN_ERRORS = 30
    EB_N0 = np.arange(0.0, 6.0, 0.5)

    save_frozen_set_info([256, 512], None, DESIGN_EBN0, 'results/frozen_sets.txt')

    # 实验一：SC
    all_sc = {}
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0
        results = run_simulation(
            N, K, EB_N0,
            lambda llr, f=fb: (sc_decode(llr, f), None),
            'sc', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        all_sc[f'SC, N={N}'] = results
        save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

    plot_bler_curves(all_sc, 'SC BLER (quick)', 'results/fig1_sc_bler.png',
                     find_capacity_limit(RATE))

    # 实验二：SCL N=512
    N = 512
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    fb = np.ones(N, dtype=int)
    fb[info_idx] = 0
    all_scl = {}
    for L in [1, 2, 4, 8]:
        dec = SCLDecoder(N, fb, list_size=max(L, 1), crc_length=0)
        label = 'SC (L=1)' if L == 1 else f'SCL (L={L})'
        results = run_simulation(
            N, K, EB_N0,
            lambda llr, d=dec: (d.decode(llr)[0], None),
            'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        all_scl[label] = results
        if L == 4:
            save_results_csv(results, f'results/exp2_scl_N{N}_R0.5.csv')

    cascl = SCLDecoder(N, fb, list_size=8, crc_length=8)
    r_cascl = run_simulation(
        N, K, EB_N0,
        lambda llr: (cascl.decode(llr)[0], None),
        'scl', MAX_FRAMES, MIN_ERRORS, crc_length=8, info_indices=info_idx,
    )
    all_scl['CA-SCL (L=8)'] = r_cascl
    plot_bler_curves(all_scl, 'SCL BLER (quick)', 'results/fig2_scl_bler.png',
                     find_capacity_limit(RATE))

    # 实验三：BP
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0
        bp = BPDecoder(N, fb, max_iter=50)
        all3 = {}
        all3['SC'] = run_simulation(
            N, K, EB_N0,
            lambda llr, f=fb: (sc_decode(llr, f), None),
            'sc', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        scl = SCLDecoder(N, fb, list_size=4)
        all3['SCL (L=4)'] = run_simulation(
            N, K, EB_N0,
            lambda llr, d=scl: (d.decode(llr)[0], None),
            'scl', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        all3['BP'] = run_simulation(
            N, K, EB_N0,
            lambda llr, d=bp: d.decode(llr),
            'bp', MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        save_results_csv(all3['BP'], f'results/exp3_bp_N{N}_R0.5.csv')
        plot_bler_curves(all3, f'Compare N={N}', f'results/fig3_bp_N{N}_bler.png',
                         find_capacity_limit(RATE))

    print("快速仿真完成。")


if __name__ == '__main__':
    main()
