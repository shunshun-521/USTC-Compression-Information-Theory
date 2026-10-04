"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from encoder import bit_reversal_permutation, polar_encode
from simulation import apply_env_overrides, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def _unit_tests():
    u = np.array([1, 0, 1, 1])
    G = np.eye(4, dtype=int)
    F = np.array([[1, 0], [1, 1]], int)
    G = np.kron(F, F)
    br = bit_reversal_permutation(4)
    G = (np.eye(4)[br] @ G) % 2
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"


def main():
    _unit_tests()
    os.makedirs("results", exist_ok=True)

    N_LIST = [256, 512, 1024]
    if os.environ.get("POLAR_SKIP_N1024", "").lower() in ("1", "true", "yes"):
        N_LIST = [256, 512]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    default_eb = np.arange(0.0, 5.5, 0.25)
    MAX_FRAMES, MIN_ERRORS, EB_N0_RANGE = apply_env_overrides(100000, 100, default_eb)

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
            info_idx=info_idx,
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
