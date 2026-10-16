"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode
from simulation import apply_quick_env_defaults, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs("results", exist_ok=True)


def _unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u2 = np.zeros(N, dtype=int)
        u2[info] = payload
        x2 = polar_encode(u2)
        y = bpsk_modulate(x2) + rng.normal(0, sigma, N)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if not np.array_equal(uh[info], payload):
            err += 1
    assert err == 0, f"SC 无损校验失败: {err}/100 帧错误"


if __name__ == "__main__":
    _unit_tests()

    quick = apply_quick_env_defaults()
    N_LIST = quick["n_list_sc"] if quick else [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_FRAMES = quick["max_frames"] if quick else 100000
    MIN_ERRORS = quick["min_errors"] if quick else 100
    EB_N0_RANGE = quick["eb_n0_range"] if quick else np.arange(0.0, 5.5, 0.25)

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'='*60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'='*60}")
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
