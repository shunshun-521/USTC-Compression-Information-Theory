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
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)


def run_unit_tests():
    """数值正确性校验。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(1):
        G = np.kron(G, F) % 2
    expected = (u @ G) % 2
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码校验失败: {errors}/100 帧错误"

    from decoder_scl import SCLDecoder

    payload = rng.integers(0, 2, size=K)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = payload
    x = polar_encode(u_sent)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "单路径 SCL 应等价于 SC"

    print("单元测试通过。")


if __name__ == "__main__":
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
