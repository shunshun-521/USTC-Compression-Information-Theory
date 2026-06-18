"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import quick_mode, run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), "results")

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

if quick_mode():
    N_LIST = [256]
    MAX_FRAMES = 500
    MIN_ERRORS = 10
    EB_N0_RANGE = np.arange(2.0, 3.5, 0.5)


def run_unit_tests():
    from verify import run_all

    run_all()


if __name__ == "__main__":
    run_unit_tests()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, os.path.join(RESULTS, "frozen_sets.txt"))

    all_results = {}

    for N in N_LIST:
        K = int(N * RATE)
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'=' * 60}")

        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits.astype(bool)), None

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
        save_results_csv(results, os.path.join(RESULTS, f"exp1_sc_N{N}_R0.5.csv"))

    shannon_db = find_capacity_limit(RATE)
    print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path=os.path.join(RESULTS, "fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")
