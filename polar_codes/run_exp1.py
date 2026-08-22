# run_exp1.py
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

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    """数值正确性校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors}/100 帧错误"

    u_hat_r = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat[info_idx], u_hat_r[info_idx]), "递归 SC 与非递归 SC 不一致"

    print("单元测试通过。")


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    run_unit_tests()

    n_list = [256, 512, 1024]
    if os.environ.get("POLAR_QUICK", "0") == "1":
        n_list = [256]

    rate = 0.5
    design_ebn0 = 2.5
    max_frames = 100000
    min_errors = 100
    eb_n0_range = np.arange(0.0, 5.5, 0.25)
    if os.environ.get("POLAR_QUICK", "0") == "1":
        eb_n0_range = np.arange(1.0, 3.5, 0.5)
        max_frames = 2000
        min_errors = 20

    save_frozen_set_info(n_list, None, design_ebn0, os.path.join(results_dir, "frozen_sets.txt"))

    all_results = {}

    for n in n_list:
        k = n // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={n}, K={k}, R={rate}")
        print(f"{'=' * 60}")

        info_idx, frozen_idx, llr_means = ga_construction(n, k, design_ebn0)
        frozen_bits = np.ones(n, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, _frozen=frozen_bits):
            return sc_decode(llr_ch, _frozen), None

        results = run_simulation(
            N=n,
            K=k,
            eb_n0_db_list=eb_n0_range,
            decoder=decoder,
            decoder_type="sc",
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
            verbose=True,
        )

        label = f"SC, N={n}, K={k}"
        all_results[label] = results
        save_results_csv(results, os.path.join(results_dir, f"exp1_sc_N{n}_R0.5.csv"))

    shannon_db = find_capacity_limit(rate)
    print(f"\nBPSK 信道容量限（R={rate}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={rate})",
        save_path=os.path.join(results_dir, "fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")


if __name__ == "__main__":
    main()
