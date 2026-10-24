# run_exp1.py
"""实验一：SC 译码基础仿真"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    skip_n1024 = os.environ.get("POLAR_SKIP_N1024", "0") == "1"

    n_list = [256, 512, 1024]
    if skip_n1024 or quick:
        n_list = [n for n in n_list if n <= 512]
    if quick:
        n_list = [256]

    rate = 0.5
    design_ebn0 = 2.5
    max_frames = 2000 if quick else 100000
    min_errors = 20 if quick else 100
    eb_n0_range = (
        np.arange(1.0, 4.0, 0.5) if quick else np.arange(0.0, 5.5, 0.25)
    )

    save_frozen_set_info(
        n_list, None, design_ebn0, os.path.join(RESULTS_DIR, "frozen_sets.txt")
    )

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
            verbose=True,
        )

        label = f"SC, N={N}, K={K}"
        all_results[label] = results
        save_results_csv(results, os.path.join(RESULTS_DIR, f"exp1_sc_N{N}_R0.5.csv"))

    shannon_db = find_capacity_limit(rate)
    print(f"\nBPSK 信道容量限（R={rate}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={rate})",
        save_path=os.path.join(RESULTS_DIR, "fig1_sc_bler.png"),
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")


if __name__ == "__main__":
    from validate import main as run_validate

    run_validate()
    main()
