#!/usr/bin/env python3
"""实验二：SCL 译码及 CRC 辅助"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit
import matplotlib.pyplot as plt
from run_exp1 import run_unit_tests


def main():
    os.makedirs('results', exist_ok=True)
    run_unit_tests()

    N = 512
    RATE = 0.5
    K = N // 2
    DESIGN_EBN0 = 2.5
    CRC_LENGTH = 8
    L_LIST = [2, 4, 8]
    MAX_FRAMES = 100000
    MIN_ERRORS = 100
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print("\nSC 基线 (L=1)")
    results_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_decoder, 'sc', MAX_FRAMES, MIN_ERRORS,
        info_indices=info_idx, verbose=True,
    )
    all_results['SC (L=1)'] = results_sc
    save_results_csv(results_sc, f'results/exp2_sc_N{N}_R0.5.csv')

    for L in L_LIST:
        print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

        def make_scl_decoder(llr_ch, _L=L):
            u_hat, _ = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
            return u_hat, None

        results = run_simulation(
            N, K, EB_N0_RANGE, make_scl_decoder, 'scl',
            MAX_FRAMES, MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        all_results[f'SCL (L={L})'] = results
        save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

    print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")

    def cascl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
        return u_hat, None

    results_cascl = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, 'scl',
        MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
        info_indices=info_idx, verbose=True,
    )
    all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = results_cascl
    save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(all_results, f'SCL vs SC BLER (N={N}, R={RATE})',
                     'results/fig2_scl_bler.png', shannon_limit_db=shannon_db)

    labels = list(all_results.keys())
    avg_times = [np.mean([r['avg_decode_time'] for r in v]) * 1000 for v in all_results.values()]
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

    u_test = np.zeros(N, int)
    u_test[info_idx[:4]] = [1, 0, 1, 1]
    llr_test = compute_llr(bpsk_modulate(polar_encode(u_test)), eb_n0_to_sigma(5.0, RATE))
    uh_sc = sc_decode(llr_test, frozen_bits)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(uh_sc, uh_scl), "SCL L=1 不等价于 SC"
    print("\n实验二完成。")


if __name__ == '__main__':
    main()
