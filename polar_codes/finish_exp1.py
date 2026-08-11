#!/usr/bin/env python3
"""补全未完成的实验并生成图表"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def finish_exp1():
    """仅重跑 N=1024 并更新 fig1"""
    N_LIST = [256, 512, 1024]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)
    all_results = {}

    for N in N_LIST:
        K = N // 2
        csv_path = f"results/exp1_sc_N{N}_R0.5.csv"
        if N < 1024 and os.path.exists(csv_path):
            from utils import load_results_csv
            all_results[f"SC, N={N}, K={K}"] = load_results_csv(csv_path)
            print(f"Loaded existing {csv_path}")
            continue

        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        print(f"Running SC N={N}")
        results = run_simulation(
            N, K, EB_N0_RANGE, decoder, "sc",
            max_frames=10000, min_errors=100,
            info_indices=info_idx, verbose=True,
        )
        all_results[f"SC, N={N}, K={K}"] = results
        save_results_csv(results, csv_path)

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path="results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )
    print("exp1 finished")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    finish_exp1()
