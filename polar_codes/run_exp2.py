#!/usr/bin/env python3
"""实验二：SCL 译码及 CRC 辅助"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit


def run_unit_tests():
    encoded = crc_encode(np.array([1, 0, 1, 1]), 8)
    from decoder_scl import crc_check
    assert crc_check(encoded, 8)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), 0.01, rng), 0.01)
        assert np.array_equal(sc_decode(llr, frozen_bits),
                             SCLDecoder(N, frozen_bits, list_size=1).decode(llr)[0])
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    N, RATE, K = 512, 0.5, 256
    CRC_LENGTH = 8
    L_LIST = [2, 4, 8]
    MAX_FRAMES, MIN_ERRORS = 100000, 100
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    results_sc = run_simulation(
        N, K, EB_N0_RANGE,
        lambda llr: (sc_decode(llr, frozen_bits), None),
        "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in L_LIST:
        print(f"\nSCL 仿真: N={N}, K={K}, L={L}")
        results = run_simulation(
            N, K, EB_N0_RANGE,
            lambda llr, _L=L: (SCLDecoder(N, frozen_bits, list_size=_L).decode(llr)[0], None),
            "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx,
        )
        all_results[f"SCL (L={L})"] = results
        save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")
    results_cascl = run_simulation(
        N, K, EB_N0_RANGE,
        lambda llr: (SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH).decode(llr)[0], None),
        "scl", MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH, info_indices=info_idx,
    )
    all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
    save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(all_results, f"SCL vs SC BLER (N={N}, R={RATE})",
                     "results/fig2_scl_bler.png", shannon_limit_db=shannon_db)

    labels = list(all_results.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title(f"Decoding Time vs List Size (N={N})")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()
    print("\n实验二完成。")


if __name__ == "__main__":
    main()
