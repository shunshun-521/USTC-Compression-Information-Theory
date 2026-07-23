"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decode
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """数值正确性校验"""
    # 编码器校验：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    from encoder import build_generator_matrix
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    # SC 译码校验（高信噪比无损）
    from decoder_sc import verify_sc_decoders
    verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=15.0)

    # 路径度量校验：L=1 SCL 等价于 SC
    from decoder_scl import verify_scl_equals_sc
    verify_scl_equals_sc(N=64, K=32, eb_n0_db=15.0)

    print("所有单元测试通过。")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    run_unit_tests()

    N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = 100000
    MIN_ERRORS = 100
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, 'results/frozen_sets.txt')

    all_results = {}

    for N in N_LIST:
        K = N // 2
        print(f"\n{'='*60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'='*60}")

        info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        results = run_simulation(
            N=N, K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            info_indices=info_idx,
            decoder_type='sc',
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            verbose=True
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
        shannon_limit_db=shannon_db
    )
    print("\n实验一完成。结果保存至 results/ 目录。")
