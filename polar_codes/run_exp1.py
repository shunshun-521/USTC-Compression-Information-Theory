"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, verify_sc_decoders
from encoder import polar_encode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)


def _unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    assert verify_sc_decoders(N=64, K=32, num_frames=50, eb_n0_db=10.0), "SC 校验失败"
    print("单元测试通过。")


def main():
    _unit_tests()
    os.makedirs("results", exist_ok=True)

    if os.environ.get("POLAR_QUICK"):
        N_LIST = [256]
    else:
        N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
    MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
    EB_N0_RANGE = (
        np.arange(1.0, 5.5, 0.5)
        if os.environ.get("POLAR_QUICK")
        else np.arange(0.0, 5.5, 0.25)
    )

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, _fb=frozen_bits):
            return sc_decode(llr_ch, _fb), None

        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            info_idx=info_idx,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
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
