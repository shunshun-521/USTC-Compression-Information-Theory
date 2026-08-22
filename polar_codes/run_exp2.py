# run_exp2.py
"""
实验二：SCL 译码及 CRC 辅助
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv
from run_exp1 import run_unit_tests

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
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

    if os.environ.get("POLAR_QUICK", "0") == "1":
        eb_n0_range = np.arange(1.5, 3.5, 0.5)
        max_frames = 1000
        min_errors = 15
        l_list = [2, 4]

    info_idx, _, _ = ga_construction(n, k, design_ebn0)
    frozen_bits = np.ones(n, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print("SC 基线 (L=1)")
    results_sc = run_simulation(
        n, k, eb_n0_range, sc_decoder, "sc", max_frames, min_errors, info_indices=info_idx
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, os.path.join(RESULTS_DIR, f"exp2_sc_N{n}_R0.5.csv"))

    for list_size in l_list:
        print(f"\nSCL 仿真: N={n}, K={k}, L={list_size}")

        def scl_decoder(llr_ch, _L=list_size):
            u_hat, _ = SCLDecoder(n, frozen_bits, list_size=_L, crc_length=0).decode(llr_ch)
            return u_hat, None

        results = run_simulation(
            n, k, eb_n0_range, scl_decoder, "scl", max_frames, min_errors, info_indices=info_idx
        )
        all_results[f"SCL (L={list_size})"] = results
        save_results_csv(results, os.path.join(RESULTS_DIR, f"exp2_scl_L{list_size}_N{n}_R0.5.csv"))

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
    )
    all_results[f"CA-SCL (L=8, CRC={crc_length})"] = results_cascl
    save_results_csv(results_cascl, os.path.join(RESULTS_DIR, f"exp2_cascl_L8_N{n}_R0.5.csv"))

    llr_test = np.random.randn(n)
    u_sc, _ = sc_decoder(llr_test)
    u_scl, _ = SCLDecoder(n, frozen_bits, list_size=1, crc_length=0).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "单路径 SCL 与 SC 不一致"

    shannon_db = find_capacity_limit(rate)
    plot_bler_curves(
        all_results,
        f"SCL vs SC BLER (N={n}, R={rate})",
        os.path.join(RESULTS_DIR, "fig2_scl_bler.png"),
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
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.png"), dpi=150)
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_decode_time.pdf"))
    plt.close()

    print("\n实验二完成。")


if __name__ == "__main__":
    main()
