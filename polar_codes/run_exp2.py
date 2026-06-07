"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check, scl_equals_sc
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 0, 1, 1]), 8)
    assert crc_check(bits, 8), "CRC 校验失败"

    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * bpsk_modulate(x)
        assert scl_equals_sc(N, frozen, llr), "L=1 SCL 与 SC 不等价"
    print("单元测试通过。")


def _sim_params():
    quick = os.environ.get('POLAR_QUICK', '0') == '1'
    return {
        'max_frames': int(os.environ.get('POLAR_MAX_FRAMES', '300' if quick else '100000')),
        'min_errors': int(os.environ.get('POLAR_MIN_ERRORS', '10' if quick else '100')),
        'eb_n0_range': np.arange(2.0, 4.0, 0.5) if quick else np.arange(1.0, 5.5, 0.25),
        'n': 128 if quick else 512,
        'l_list': [2, 4] if quick else [2, 4, 8],
    }


def main():
    run_unit_tests()
    os.makedirs('results', exist_ok=True)
    p = _sim_params()

    N = p['n']
    RATE = 0.5
    K = N // 2
    DESIGN_EBN0 = 2.5
    CRC_LENGTH = 8
    L_LIST = p['l_list']

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print(f"\nSC 基线 (L=1), N={N}")
    results_sc = run_simulation(
        N, K, p['eb_n0_range'], sc_decoder, 'sc',
        p['max_frames'], p['min_errors'],
        info_indices=info_idx, verbose=True,
    )
    all_results['SC (L=1)'] = results_sc
    save_results_csv(results_sc, f'results/exp2_sc_N{N}_R0.5.csv')

    for L in L_LIST:
        print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

        def scl_decoder(llr_ch, _L=L):
            u_hat, pm = SCLDecoder(N, frozen_bits, list_size=_L, crc_length=0).decode(llr_ch)
            return u_hat, None

        results = run_simulation(
            N, K, p['eb_n0_range'], scl_decoder, 'scl',
            p['max_frames'], p['min_errors'],
            info_indices=info_idx, verbose=True,
        )
        all_results[f'SCL (L={L})'] = results
        save_results_csv(results, f'results/exp2_scl_L{L}_N{N}_R0.5.csv')

    print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")

    def cascl_decoder(llr_ch):
        u_hat, pm = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr_ch)
        return u_hat, None

    results_cascl = run_simulation(
        N, K, p['eb_n0_range'], cascl_decoder, 'scl',
        p['max_frames'], p['min_errors'],
        crc_length=CRC_LENGTH, info_indices=info_idx, verbose=True,
    )
    all_results[f'CA-SCL (L=8, CRC={CRC_LENGTH})'] = results_cascl
    save_results_csv(results_cascl, f'results/exp2_cascl_L8_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results, f'SCL vs SC BLER (N={N}, R={RATE})',
        'results/fig2_scl_bler.png', shannon_limit_db=shannon_db,
    )

    if plt is not None:
        labels = list(all_results.keys())
        avg_times = [
            np.mean([r['avg_decode_time'] for r in v]) * 1000
            for v in all_results.values()
        ]
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

    print("\n实验二完成。")


if __name__ == '__main__':
    main()
