#!/usr/bin/env python3
"""补全未完成的仿真结果（exp1 N512、exp2 全部、exp3 全部）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv, load_results_csv

os.makedirs('results', exist_ok=True)

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)


def run_exp1_n512():
    N = 512
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print('补全 exp1 N=512 ...')
    results = run_simulation(
        N, K, EB_N0_RANGE, decoder, 'sc', MAX_FRAMES, MIN_ERRORS,
        info_idx=info_idx, frozen_bits=frozen_bits,
    )
    save_results_csv(results, 'results/exp1_sc_N512_R0.5.csv')

    all_results = {}
    for n in [256, 512, 1024]:
        path = f'results/exp1_sc_N{n}_R0.5.csv'
        if os.path.exists(path):
            all_results[f'SC, N={n}, K={n//2}'] = load_results_csv(path)
    plot_bler_curves(
        all_results, f'SC Decoder BLER vs Eb/N0 (R={RATE})',
        'results/fig1_sc_bler.png', find_capacity_limit(RATE),
    )


def run_exp2_full():
    N = 512
    K = N // 2
    CRC_LENGTH = 8
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print('exp2 SC ...')
    results_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_decoder, 'sc', MAX_FRAMES, MIN_ERRORS,
        info_idx=info_idx, frozen_bits=frozen_bits,
    )
    all_results['SC (L=1)'] = results_sc
    save_results_csv(results_sc, f'results/exp2_sc_N{N}_R0.5.csv')

    for L in [2, 4, 8]:
        print(f'exp2 SCL L={L} ...')

        def make_scl(_L=L):
            def scl_decoder(llr_ch):
                u, pm = SCLDecoder(N, frozen_bits, list_size=_L).decode(llr_ch)
                return u, None
            return scl_decoder

        results = run_simulation(
            N, K, EB_N0_RANGE, make_scl(), 'scl', MAX_FRAMES, MIN_ERRORS,
            info_idx=info_idx, frozen_bits=frozen_bits,
        )
        all_results[f'SCL (L={L})'] = results
        save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')
        if L == 8:
            save_results_csv(results, f'results/exp2_scl_N{N}_R0.5.csv')

    print('exp2 CA-SCL ...')

    def cascl_decoder(llr_ch):
        u, pm = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
        return u, None

    results_cascl = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, 'scl', MAX_FRAMES, MIN_ERRORS,
        crc_length=CRC_LENGTH, info_idx=info_idx, frozen_bits=frozen_bits,
    )
    all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = results_cascl
    save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results, f'SCL vs SC BLER (N={N}, R={RATE})',
        'results/fig2_scl_bler.png', shannon_db,
    )
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


def run_exp3_full():
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f'exp3 N={N} SC ...')
        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, 'sc', MAX_FRAMES, MIN_ERRORS,
            info_idx=info_idx, frozen_bits=frozen_bits,
        )
        all_results['SC'] = r_sc
        save_results_csv(r_sc, f'results/exp3_sc_N{N}_R0.5.csv')

        def scl_d(llr_ch):
            u, pm = SCLDecoder(N, frozen_bits, list_size=4).decode(llr_ch)
            return u, None

        print(f'exp3 N={N} SCL ...')
        r_scl = run_simulation(
            N, K, EB_N0_RANGE, scl_d, 'scl', MAX_FRAMES, MIN_ERRORS,
            info_idx=info_idx, frozen_bits=frozen_bits,
        )
        all_results['SCL (L=4)'] = r_scl
        save_results_csv(r_scl, f'results/exp3_scl_N{N}_R0.5.csv')

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=50)

        def bp_d(llr_ch):
            u, iters = bp_decoder.decode(llr_ch)
            return u, iters

        print(f'exp3 N={N} BP ...')
        r_bp = run_simulation(
            N, K, EB_N0_RANGE, bp_d, 'bp', MAX_FRAMES, MIN_ERRORS,
            info_idx=info_idx, frozen_bits=frozen_bits,
        )
        all_results['BP (max_iter=50)'] = r_bp
        save_results_csv(r_bp, f'results/exp3_bp_N{N}_R0.5.csv')

        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results, f'SC vs SCL vs BP (N={N}, R={RATE})',
            f'results/fig3_bp_N{N}_bler.png', shannon_db,
        )
        eb_vals = [r['eb_n0_db'] for r in r_bp]
        iters = [r['avg_iters'] for r in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_vals, iters, 'o-', color='purple')
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Avg Iterations')
        ax.set_title(f'BP Average Iterations (N={N}, max_iter=50)')
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'results/fig3_bp_N{N}_iters.png', dpi=150)
        plt.savefig(f'results/fig3_bp_N{N}_iters.pdf')
        plt.close()


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), [1, 0, 1, 1])
    run_exp1_n512()
    run_exp2_full()
    run_exp3_full()
    print('全部补全完成。')
