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
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    n_list = [256, 512] if quick else [256, 512, 1024]
    rate = 0.5
    design_ebn0 = 2.5
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "5000" if quick else "100000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "20" if quick else "100"))
    eb_n0_range = np.arange(6.0, 10.5, 1.0) if quick else np.arange(5.0, 12.5, 0.5)

    save_frozen_set_info(n_list, None, design_ebn0, "results/frozen_sets.txt")

    all_results = {}
    for N in n_list:
        K = N // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={rate}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(N, K, design_ebn0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, fb=frozen_bits):
            return sc_decode(llr_ch, fb), None

        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=eb_n0_range,
            decoder=decoder,
            decoder_type="sc",
            max_frames=max_frames,
            min_errors=min_errors,
            info_indices=info_idx,
            frozen_bits=frozen_bits,
            design_eb_n0_db=design_ebn0,
        )

        label = f"SC, N={N}, K={K}"
        all_results[label] = results
        save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

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
