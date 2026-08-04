"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """模块正确性校验。"""
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u, u_hat)

    print("单元测试通过。")


def main():
    os.makedirs('results', exist_ok=True)
    run_unit_tests()

    n_list = [256, 512, 1024]
    rate = 0.5
    design_ebn0 = 2.5
    max_frames = 100000
    min_errors = 100
    eb_n0_range = np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(n_list, None, design_ebn0, 'results/frozen_sets.txt')

    all_results = {}

    for n in n_list:
        k = n // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={n}, K={k}, R={rate}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(n, k, design_ebn0)
        frozen_bits = np.ones(n, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, fb=frozen_bits):
            return sc_decode(llr_ch, fb)

        results = run_simulation(
            N=n, K=k,
            eb_n0_db_list=eb_n0_range,
            decoder=decoder,
            decoder_type='sc',
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
            verbose=True,
        )

        label = f'SC, N={n}, K={k}'
        all_results[label] = results
        save_results_csv(results, f'results/exp1_sc_N{n}_R0.5.csv')

    shannon_db = find_capacity_limit(rate)
    print(f"\nBPSK 信道容量限（R={rate}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f'SC Decoder BLER vs Eb/N0 (R={rate})',
        save_path='results/fig1_sc_bler.png',
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")


if __name__ == '__main__':
    main()
