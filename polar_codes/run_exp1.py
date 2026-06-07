"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """模块正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u_sent[info_idx]), "SC 高信噪比译码失败"
    print("单元测试通过。")


def _sim_params():
    quick = os.environ.get('POLAR_QUICK', '0') == '1'
    return {
        'max_frames': int(os.environ.get('POLAR_MAX_FRAMES', '500' if quick else '100000')),
        'min_errors': int(os.environ.get('POLAR_MIN_ERRORS', '10' if quick else '100')),
        'eb_n0_range': np.arange(1.0, 4.0, 0.5) if quick else np.arange(0.0, 5.5, 0.25),
        'n_list': [64, 128] if quick else [256, 512, 1024],
    }


def main():
    run_unit_tests()
    os.makedirs('results', exist_ok=True)
    params = _sim_params()

    N_LIST = params['n_list']
    RATE = 0.5
    DESIGN_EBN0 = 2.5

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, 'results/frozen_sets.txt')

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        results = run_simulation(
            N=N, K=K,
            eb_n0_db_list=params['eb_n0_range'],
            decoder=decoder,
            decoder_type='sc',
            max_frames=params['max_frames'],
            min_errors=params['min_errors'],
            info_indices=info_idx,
            verbose=True,
        )

        label = f'SC, N={N}, K={K}'
        all_results[label] = results
        save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")
    plot_bler_curves(
        all_results,
        title=f'SC Decoder BLER vs Eb/N0 (R={RATE})',
        save_path='results/fig1_sc_bler.png',
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")


if __name__ == '__main__':
    main()
