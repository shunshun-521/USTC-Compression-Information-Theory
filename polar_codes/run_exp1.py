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
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """数值正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, K)
        x_test = polar_encode(u_test)
        y_test = bpsk_modulate(x_test) + rng.normal(0, sigma, N)
        llr_test = compute_llr(y_test, sigma)
        u_hat = sc_decode(llr_test, frozen_bits)
        if not np.array_equal(u_hat, u_test):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码校验失败: {sc_errors}/100 帧错误"

    from decoder_scl import SCLDecoder
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    rng = np.random.default_rng(1)
    scl_errors = 0
    for _ in range(50):
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, K)
        llr_test = compute_llr(bpsk_modulate(polar_encode(u_test)) + rng.normal(0, sigma, N), sigma)
        u_scl, _ = scl.decode(llr_test)
        if not np.array_equal(u_scl, u_test):
            scl_errors += 1
    assert scl_errors == 0, f"L=1 SCL 应等价于 SC: {scl_errors}/50 帧错误"
    print("单元测试全部通过。")


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
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, llr_means = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N=N, K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type='sc',
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
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
