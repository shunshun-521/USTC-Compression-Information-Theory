"""
实验一：SC 译码基础仿真
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
    reorder_llr_for_decoder,
)
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    """模块单元测试"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 8, 4
    info_idx = np.array([0, 1, 3, 7])
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = reorder_llr_for_decoder(
            compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N)), N
        )
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh, u), "SC 译码在 N=8 下应无错误"

    u0 = np.zeros(N, dtype=int)
    u0[info_idx] = rng.integers(0, 2, K)
    llr0 = reorder_llr_for_decoder(
        compute_llr(bpsk_modulate(polar_encode(u0)), eb_n0_to_sigma(10.0, K / N)), N
    )
    assert np.array_equal(sc_decode_recursive(llr0, frozen.astype(bool)), u0)
    assert np.array_equal(sc_decode(llr0, frozen), sc_decode_recursive(llr0, frozen.astype(bool)))
    print("单元测试通过。")


def main():
    run_unit_tests()

    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", "100000" if not quick else "5000"))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", "100" if not quick else "20"))

    os.makedirs("results", exist_ok=True)

    N_LIST = [256, 512, 1024] if not quick else [256, 512]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
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
            info_indices=info_idx,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=max_frames,
            min_errors=min_errors,
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
