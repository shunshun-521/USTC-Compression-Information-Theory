"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DESIGN_EBN0, EB_N0_MAX, EB_N0_MIN, EB_N0_STEP, MAX_FRAMES, MIN_ERRORS
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    from validate import validate_sc_lossless, validate_scl_equals_sc

    validate_sc_lossless()
    validate_scl_equals_sc()
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n_list = [int(x) for x in os.environ.get("POLAR_N_LIST", "256,512,1024").split(",")]
    if os.environ.get("POLAR_SKIP_N1024", "0") == "1":
        n_list = [n for n in n_list if n != 1024]

    rate = 0.5
    eb_n0_range = np.arange(EB_N0_MIN, EB_N0_MAX + 1e-9, EB_N0_STEP)

    save_frozen_set_info(n_list, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}

    for n in n_list:
        k = n // 2
        print(f"\n{'=' * 60}\nSC 仿真: N={n}, K={k}, R={rate}\n{'=' * 60}")

        info_idx, _, _ = ga_construction(n, k, DESIGN_EBN0)
        frozen_bits = np.ones(n, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, _fb=frozen_bits):
            return sc_decode(llr_ch, _fb), None

        results = run_simulation(
            N=n,
            K=k,
            eb_n0_db_list=eb_n0_range,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            frozen_bits=frozen_bits,
            info_indices=info_idx,
            verbose=True,
        )

        label = f"SC, N={n}, K={k}"
        all_results[label] = results
        save_results_csv(results, f"results/exp1_sc_N{n}_R0.5.csv")

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
