"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
- 仿真并绘制 BLER-Eb/N0 曲线
- 添加 BPSK 信道容量限
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
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

    print("运行单元测试...")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("  编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info_bits
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 帧错误"
    print("  SC 译码校验通过")

    llr_test = np.array([1.5, -0.5, 2.0, -1.0])
    frozen_test = np.array([0, 1, 0, 1])
    u_rec = sc_decode_recursive(llr_test, frozen_test)
    u_nr = sc_decode(llr_test, frozen_test)
    assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 结果不一致"
    print("  递归/非递归 SC 一致性校验通过")

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"  GA N=8: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  GA N=256 前20个 info_indices: {info256[:20]}")
    print("单元测试全部通过。\n")


os.makedirs("results", exist_ok=True)

_QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

N_LIST = [64, 128] if _QUICK else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if _QUICK else 100000
MIN_ERRORS = 5 if _QUICK else 100
EB_N0_RANGE = np.arange(1.0, 3.5, 0.5) if _QUICK else np.arange(0.0, 5.5, 0.25)

if __name__ == "__main__":
    run_unit_tests()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

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
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            verbose=True,
            info_indices=info_idx,
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
