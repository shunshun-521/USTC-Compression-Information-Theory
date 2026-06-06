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

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decode,
)
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    """数值正确性校验"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    assert np.array_equal(x, (u @ g) % 2), f"编码器错误: {x}"

    n, k = 64, 32
    info_idx, _, _ = ga_construction(n, k, 2.5)
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, k / n)
    errors = 0
    for _ in range(100):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        llr = reorder_llr_for_decode(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng), sigma),
            n,
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u_test):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100"
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n_list = [256, 512, 1024]
    rate = 0.5
    design_ebn0 = 2.5
    max_frames = 100000
    min_errors = 100
    eb_n0_range = np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(n_list, None, design_ebn0, "results/frozen_sets.txt")

    all_results = {}
    for n_val in n_list:
        k_val = n_val // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={n_val}, K={k_val}, R={rate}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(n_val, k_val, design_ebn0)
        frozen_bits = np.ones(n_val, dtype=bool)
        frozen_bits[info_idx] = False

        def decoder(llr_ch, _fb=frozen_bits):
            return sc_decode(llr_ch, _fb), None

        results = run_simulation(
            N=n_val,
            K=k_val,
            eb_n0_db_list=eb_n0_range,
            decoder=decoder,
            decoder_type="sc",
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
            verbose=True,
        )

        label = f"SC, N={n_val}, K={k_val}"
        all_results[label] = results
        save_results_csv(results, f"results/exp1_sc_N{n_val}_R0.5.csv")

    shannon_db = find_capacity_limit(rate)
    print(f"\nBPSK 信道容量限（R={rate}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={rate})",
        save_path="results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")


if __name__ == "__main__":
    main()
