"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    from encoder import build_generator_matrix

    x = polar_encode(u)
    assert np.array_equal(x, (u @ build_generator_matrix(4)) % 2)

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    fb = np.ones(N, dtype=int)
    fb[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(7)
    for _ in range(50):
        u64 = np.zeros(N, dtype=int)
        u64[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u64)), sigma, rng), sigma
        )
        assert np.array_equal(
            sc_decode(llr, fb),
            SCLDecoder(N, fb.astype(bool), list_size=1).decode(llr)[0],
        )
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8
L_LIST = [2, 4, 8]
MAX_FRAMES_SC = 50000
MAX_FRAMES_SCL = {2: 2000, 4: 800, 8: 200}
MIN_ERRORS = 100
MIN_ERRORS_SCL = 30
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

if __name__ == "__main__":
    run_unit_tests()

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print("\nSC 基线 (L=1)")
    results_sc = run_simulation(
        N, K, EB_N0_RANGE, sc_decoder, "sc",
        MAX_FRAMES_SC, MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in L_LIST:
        print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

        def scl_decoder(llr_ch, _L=L):
            u_hat, pm = SCLDecoder(
                N, frozen_bits.astype(bool), list_size=_L, crc_length=0
            ).decode(llr_ch)
            return u_hat, None

        max_f = MAX_FRAMES_SCL.get(_L, 800)
        results = run_simulation(
            N, K, EB_N0_RANGE, scl_decoder, "scl",
            max_f, MIN_ERRORS_SCL, info_indices=info_idx, verbose=True,
        )
        all_results[f"SCL (L={L})"] = results
        save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")
        if L == 4:
            save_results_csv(results, f"results/exp2_scl_N{N}_R0.5.csv")

    print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")

    def cascl_decoder(llr_ch):
        u_hat, pm = SCLDecoder(
            N, frozen_bits.astype(bool), list_size=8, crc_length=CRC_LENGTH
        ).decode(llr_ch)
        return u_hat, None

    results_cascl = run_simulation(
        N, K, EB_N0_RANGE, cascl_decoder, "scl",
        MAX_FRAMES_SCL[8], MIN_ERRORS_SCL, crc_length=CRC_LENGTH,
        info_indices=info_idx, verbose=True,
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
