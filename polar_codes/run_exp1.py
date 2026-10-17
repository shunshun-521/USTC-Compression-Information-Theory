"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def _unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # L=1 SCL 等价于 SC
    N, K = 8, 4
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    from decoder_scl import SCLDecoder

    sigma = eb_n0_to_sigma(10.0, 0.5)
    u = np.zeros(N, dtype=int)
    u[info_idx] = [1, 0, 1, 1]
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    uh, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh[info_idx], u[info_idx]), "SCL(L=1) 与 SC 路径不一致"
    print("SCL(L=1) 单帧自检通过")


def main():
    _unit_tests()

    os.makedirs("results", exist_ok=True)
    quick = os.environ.get("POLAR_QUICK", "0") == "1"

    N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = 500 if quick else 100000
    MIN_ERRORS = 20 if quick else 100
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.5 if quick else 0.25)

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")
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
