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
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from simulation import run_simulation
from utils import (
    save_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    find_capacity_limit,
)


def run_unit_tests():
    """数值正确性校验。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]])
    G = F
    for _ in range(1):
        G = np.kron(G, F)
    from encoder import bit_reversal_permutation

    rev = bit_reversal_permutation(4)
    B = np.zeros((4, 4), dtype=int)
    for i in range(4):
        B[i, rev[i]] = 1
    GN = (G @ B) % 2
    assert np.array_equal(x, (u @ GN) % 2), f"编码器错误: {x}"
    print("编码器校验通过")

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits)
        if np.any(u_hat[info_idx] != u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码校验失败: {errors}/100 帧有错误"
    print("SC 译码校验通过 (N=64, Eb/N0=10dB, 100帧)")

    # 递归与非递归一致性
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.1)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
    print("递归/非递归 SC 一致性校验通过")


def main():
    run_unit_tests()

    os.makedirs("results", exist_ok=True)

    N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = 100000
    MIN_ERRORS = 100
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

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
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            info_indices=info_idx,
            frozen_bits=frozen_bits.astype(bool),
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


if __name__ == "__main__":
    main()
