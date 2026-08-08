"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """模块正确性校验"""
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, np.dot(u, G) % 2), f"编码器错误: {x}"

    # SC 译码：极低噪声下应无错
    N, K = 64, 32
    n = int(np.log2(N))
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_full)), sigma)[br]
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u_full), "SC 译码失败"

    # SCL L=1 等价于 SC
    from decoder_scl import SCLDecoder

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_full)), sigma)[br]
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("单元测试通过。")


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
