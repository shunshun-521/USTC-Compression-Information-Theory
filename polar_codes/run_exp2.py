"""
实验二：SCL 译码及 CRC 辅助
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DESIGN_EBN0, EB_N0_MAX, EB_N0_MIN, EB_N0_STEP, MAX_FRAMES, MIN_ERRORS
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from run_exp1 import run_unit_tests
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    n = 512
    rate = 0.5
    k = n // 2
    crc_length = 8
    l_list = [int(x) for x in os.environ.get("POLAR_L_LIST", "2,4,8").split(",")]
    eb_n0_range = np.arange(max(EB_N0_MIN, 1.0), EB_N0_MAX + 1e-9, EB_N0_STEP)

    info_idx, _, _ = ga_construction(n, k, DESIGN_EBN0)
    frozen_bits = np.ones(n, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}

    if os.environ.get("POLAR_SKIP_SC", "0") != "1":
        def sc_decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        results_sc = run_simulation(
            n,
            k,
            eb_n0_range,
            sc_decoder,
            "sc",
            MAX_FRAMES,
            MIN_ERRORS,
            frozen_bits=frozen_bits,
            info_indices=info_idx,
            verbose=True,
        )
        all_results["SC (L=1)"] = results_sc
        save_results_csv(results_sc, f"results/exp2_sc_N{n}_R0.5.csv")

    if os.environ.get("POLAR_SKIP_SCL", "0") != "1":
        for l in l_list:
            print(f"\nSCL 仿真: N={n}, K={k}, L={l}")

            def scl_decoder(llr_ch, _l=l):
                u_hat, _ = SCLDecoder(n, frozen_bits, list_size=_l, crc_length=0).decode(
                    llr_ch
                )
                return u_hat, None

            results = run_simulation(
                n,
                k,
                eb_n0_range,
                scl_decoder,
                "scl",
                MAX_FRAMES,
                MIN_ERRORS,
                frozen_bits=frozen_bits,
                info_indices=info_idx,
                verbose=True,
            )
            all_results[f"SCL (L={l})"] = results
            save_results_csv(results, f"results/exp2_scl_L{l}_N{n}_R0.5.csv")

    if os.environ.get("POLAR_SKIP_CASCL", "0") != "1":
        print(f"\nCA-SCL 仿真: N={n}, K={k}, L=8, CRC={crc_length}")

        def cascl_decoder(llr_ch):
            u_hat, _ = SCLDecoder(
                n, frozen_bits, list_size=8, crc_length=crc_length
            ).decode(llr_ch)
            return u_hat, None

        results_cascl = run_simulation(
            n,
            k,
            eb_n0_range,
            cascl_decoder,
            "scl",
            MAX_FRAMES,
            MIN_ERRORS,
            crc_length=crc_length,
            frozen_bits=frozen_bits,
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
    avg_times = [
        np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()
    ]

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
