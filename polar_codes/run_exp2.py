"""
实验二：SCL 译码及 CRC 辅助
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(1)
    for _ in range(30):
        u = np.zeros(N, int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5)
        )
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    run_unit_tests()

    N = 512
    RATE = 0.5
    K = N // 2
    DESIGN_EBN0 = 2.5
    CRC_LENGTH = 8
    L_LIST = [2, 4, 8]
    MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
    MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
    EB_N0_RANGE = np.arange(2.0, 8.5, 0.25)

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print("SC 基线 (L=1)")
    results_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_decoder, "sc", MAX_FRAMES, MIN_ERRORS,
        frozen_bits=frozen_bits, info_indices=info_idx,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in L_LIST:
        print(f"\nSCL 仿真: L={L}")
        def make_scl(llr_ch, _L=L):
            u_hat, _ = SCLDecoder(N, frozen_bits, list_size=_L, crc_length=0).decode(
                llr_ch
            )
            return u_hat, None

        t0 = time.time()
        results = run_simulation(
            N, K, EB_N0_RANGE, make_scl, "scl", MAX_FRAMES, MIN_ERRORS,
            frozen_bits=frozen_bits, info_indices=info_idx,
        )
        print(f"耗时 {time.time() - t0:.1f}s")
        all_results[f"SCL (L={L})"] = results
        save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")
    if L == 4:
        save_results_csv(results, f"results/exp2_scl_N{N}_R0.5.csv")

    print(f"\nCA-SCL: L=8, CRC={CRC_LENGTH}")

    def cascl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(
            N, frozen_bits, list_size=8, crc_length=CRC_LENGTH
        ).decode(llr_ch)
        return u_hat, None

    results_cascl = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, "scl",
        MAX_FRAMES, MIN_ERRORS, crc_length=CRC_LENGTH,
        frozen_bits=frozen_bits, info_indices=info_idx,
    )
    all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = results_cascl
    save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SCL vs SC BLER (N={N}, R={RATE})",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )

    labels = list(all_results.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in v]) * 1000
        for v in all_results.values()
    ]
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
