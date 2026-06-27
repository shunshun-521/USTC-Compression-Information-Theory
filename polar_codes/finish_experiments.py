"""快速完成剩余实验（CA-SCL + 实验三）。"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, load_results_csv, plot_bler_curves, save_results_csv

MAX_FRAMES = 50000
MIN_ERRORS = 50


def finish_exp2():
    N = 512
    RATE = 0.5
    K = N // 2
    CRC_LENGTH = 8
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    all_results = {}
    for path, label in [
        ("results/exp2_sc_N512_R0.5.csv", "SC (L=1)"),
        ("results/exp2_scl_L2_N512_R0.5.csv", "SCL (L=2)"),
        ("results/exp2_scl_L4_N512_R0.5.csv", "SCL (L=4)"),
        ("results/exp2_scl_L8_N512_R0.5.csv", "SCL (L=8)"),
    ]:
        all_results[label] = load_results_csv(path)

    cascl_path = "results/exp2_cascl_L8_N512_R0.5.csv"
    if not os.path.exists(cascl_path):

        def cascl_decoder(llr_ch):
            u_hat, pm = SCLDecoder(
                N, frozen_bits, list_size=8, crc_length=CRC_LENGTH
            ).decode(llr_ch)
            return u_hat, None

        print("运行 CA-SCL 仿真...")
        results_cascl = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            cascl_decoder,
            "scl",
            MAX_FRAMES,
            MIN_ERRORS,
            crc_length=CRC_LENGTH,
            info_indices=info_idx,
        )
        save_results_csv(results_cascl, cascl_path)
    all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = load_results_csv(cascl_path)

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
    print("实验二完成。")


def finish_exp3():
    N_LIST = [256, 512]
    RATE = 0.5
    DESIGN_EBN0 = 2.5
    MAX_ITER = 50
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        all_results = {}

        def sc_d(llr_ch, _frozen=frozen_bits):
            return sc_decode(llr_ch, _frozen), None

        print(f"实验三 SC: N={N}")
        r_sc = run_simulation(
            N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr_ch, _frozen=frozen_bits):
            u, _ = SCLDecoder(N, _frozen, list_size=4).decode(llr_ch)
            return u, None

        print(f"实验三 SCL: N={N}")
        r_scl = run_simulation(
            N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)

        def bp_d(llr_ch, _bp=bp_decoder):
            u_hat, num_iters = _bp.decode(llr_ch)
            return u_hat, num_iters

        print(f"实验三 BP: N={N}")
        r_bp = run_simulation(
            N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("实验三完成。")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    finish_exp2()
    finish_exp3()
