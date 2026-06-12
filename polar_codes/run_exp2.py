"""
实验二：SCL 译码及 CRC 辅助
- 固定码长 N=512，码率 R=1/2
- 列表大小 L = 1(SC), 2, 4, 8
- CRC 辅助 CA-SCL（r=8）
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equivalent_sc
from encoder import polar_encode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_results_csv


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(12.0, 0.5)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            errors += 1
    assert errors == 0, f"SC 测试失败: {errors}/100"

    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    assert scl_equivalent_sc(llr, frozen_bits), "L=1 时 SCL 应等价于 SC"
    print("单元测试通过。")


def _sim_params():
    quick = os.environ.get("POLAR_QUICK", "0") == "1"
    return {
        "max_frames": 100 if quick else 100000,
        "min_errors": 5 if quick else 100,
        "eb_n0_range": np.arange(2.0, 3.5, 0.5) if quick else np.arange(1.0, 5.5, 0.25),
        "l_list": [2] if quick else [2, 4, 8],
        "n": 128 if quick else 512,
        "run_cascl": not quick,
    }


def main():
    os.makedirs("results", exist_ok=True)
    run_unit_tests()

    params = _sim_params()
    N = params["n"]
    RATE = 0.5
    K = N // 2
    DESIGN_EBN0 = 2.5
    CRC_LENGTH = 8

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    print("\nSC 基线 (L=1)")
    results_sc = run_simulation(
        N,
        K,
        params["eb_n0_range"],
        sc_decoder,
        "sc",
        params["max_frames"],
        params["min_errors"],
        info_indices=info_idx,
        verbose=True,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in params["l_list"]:
        print(f"\nSCL 仿真: N={N}, K={K}, L={L}")

        def scl_decoder(llr_ch, _L=L):
            u_hat, pm = SCLDecoder(
                N, frozen_bits, list_size=_L, crc_length=0
            ).decode(llr_ch)
            return u_hat, None

        results = run_simulation(
            N,
            K,
            params["eb_n0_range"],
            scl_decoder,
            "scl",
            params["max_frames"],
            params["min_errors"],
            info_indices=info_idx,
            verbose=True,
        )
        label = f"SCL (L={L})"
        all_results[label] = results
        save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    if params["run_cascl"]:
        print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")

        def cascl_decoder(llr_ch):
            u_hat, pm = SCLDecoder(
                N, frozen_bits, list_size=8, crc_length=CRC_LENGTH
            ).decode(llr_ch)
            return u_hat, None

        results_cascl = run_simulation(
            N,
            K,
            params["eb_n0_range"],
            cascl_decoder,
            "scl",
            params["max_frames"],
            params["min_errors"],
            crc_length=CRC_LENGTH,
            info_indices=info_idx,
            verbose=True,
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


if __name__ == "__main__":
    main()
