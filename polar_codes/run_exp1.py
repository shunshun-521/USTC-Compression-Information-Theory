"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    """数值正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器错误: {x} vs {xm}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = payload
        codeword = polar_encode(u_test)
        received = awgn_channel(bpsk_modulate(codeword), sigma, rng)
        llr = compute_llr(received, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u_test):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下仍有 {errors} 帧错误"

    from decoder_scl import SCLDecoder

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    codeword = polar_encode(u_test)
    llr = compute_llr(bpsk_modulate(codeword), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"

    print("实验一单元测试全部通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

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
            frozen_bits=frozen_bits,
            design_eb_n0_db=DESIGN_EBN0,
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
