"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import (
    awgn_channel,
    bpsk_modulate,
    compute_llr,
    eb_n0_to_sigma,
    reorder_llr_for_decode,
)
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), (u @ build_generator_matrix(4)) % 2)

    n, k = 64, 32
    info_idx, _, _ = ga_construction(n, k, 2.5)
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, k / n)

    for _ in range(100):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        llr = reorder_llr_for_decode(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng), sigma),
            n,
        )
        assert np.array_equal(sc_decode(llr, frozen_bits), u_test)

    scl = SCLDecoder(n, frozen_bits, list_size=1)
    for _ in range(20):
        u_test = np.zeros(n, dtype=int)
        u_test[info_idx] = rng.integers(0, 2, k)
        llr = reorder_llr_for_decode(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_test)), sigma, rng), sigma),
            n,
        )
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl)
    print("单元测试通过。")


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n = 512
    rate = 0.5
    k = n // 2
    design_ebn0 = 2.5
    crc_length = 8
    l_list = [2, 4, 8]
    max_frames = 100000
    min_errors = 100
    eb_n0_range = np.arange(1.0, 5.5, 0.25)

    info_idx, _, _ = ga_construction(n, k, design_ebn0)
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[info_idx] = False

    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results_sc = run_simulation(
        n,
        k,
        eb_n0_range,
        sc_decoder,
        "sc",
        max_frames,
        min_errors,
        info_indices=info_idx,
        verbose=True,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{n}_R0.5.csv")

    for list_size in l_list:
        print(f"\nSCL 仿真: N={n}, K={k}, L={list_size}")

        def scl_decoder(llr_ch, _l=list_size):
            u_hat, _ = SCLDecoder(n, frozen_bits, list_size=_l, crc_length=0).decode(llr_ch)
            return u_hat, None

        results = run_simulation(
            n,
            k,
            eb_n0_range,
            scl_decoder,
            "scl",
            max_frames,
            min_errors,
            info_indices=info_idx,
            verbose=True,
        )
        all_results[f"SCL (L={list_size})"] = results
        save_results_csv(results, f"results/exp2_scl_L{list_size}_N{n}_R0.5.csv")

    print(f"\nCA-SCL 仿真: N={n}, K={k}, L=8, CRC={crc_length}")

    def cascl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(n, frozen_bits, list_size=8, crc_length=crc_length).decode(llr_ch)
        return u_hat, None

    results_cascl = run_simulation(
        n,
        k,
        eb_n0_range,
        cascl_decoder,
        "scl",
        max_frames,
        min_errors,
        crc_length=crc_length,
        info_indices=info_idx,
        verbose=True,
    )
    all_results[f"CA-SCL (L=8, CRC={crc_length})"] = results_cascl
    save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{n}_R0.5.csv")

    shannon_db = find_capacity_limit(rate)
    plot_bler_curves(
        all_results,
        f"SCL vs SC BLER (N={n}, R={rate})",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )

    labels = list(all_results.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title(f"Decoding Time vs List Size (N={n})")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()

    print("\n实验二完成。")


if __name__ == "__main__":
    main()
