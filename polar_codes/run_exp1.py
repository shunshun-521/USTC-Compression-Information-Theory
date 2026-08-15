"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode, polar_encode_no_br, build_generator_matrix, bit_reversal_permutation
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_validation_tests():
    """单元测试验证各模块正确性"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    G = build_generator_matrix(4)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ G) % 2
    br = bit_reversal_permutation(4)
    x_nobr = polar_encode_no_br(u)
    assert np.array_equal(x, x_nobr[br]), f"编码器比特倒序错误: {x} vs {x_nobr[br]}"
    print(f"编码器校验通过: u={u} -> x={x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    br = bit_reversal_permutation(N)

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)[br]
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print(f"SC 无损译码校验通过 (N={N}, K={K}, 100帧)")

    from decoder_scl import SCLDecoder

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = np.random.randint(0, 2, K)
    llr_test = compute_llr(bpsk_modulate(polar_encode(u_test)), 0.001)[br]
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不等价"
    print("SCL L=1 等价 SC 校验通过")
    print()


os.makedirs("results", exist_ok=True)

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 10000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(1.0, 7.5, 0.5)

run_validation_tests()

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = int(N * RATE)
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits,
        verbose=True,
    )

    label = f"SC, N={N}, K={K}"
    all_results[label] = results
    save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

shannon_db = find_capacity_limit(RATE)
print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)

print("\n实验一完成。结果保存至 results/ 目录。")
