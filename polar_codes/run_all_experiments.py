"""
批量运行三组实验（设置 POLAR_FAST=0 使用完整仿真参数）。
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import run_exp1
import run_exp2
import run_exp3
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

FAST = os.environ.get("POLAR_FAST", "1") == "1"
if FAST:
    for mod in (run_exp1, run_exp2, run_exp3):
        mod.MAX_FRAMES = 2000
        mod.MIN_ERRORS = 30
    run_exp2.EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)
    run_exp3.EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)


def run_exp1_sim():
    run_exp1.run_validations()
    save_frozen_set_info(
        run_exp1.N_LIST, None, run_exp1.DESIGN_EBN0, "results/frozen_sets.txt"
    )
    all_results = {}
    for N in run_exp1.N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, run_exp1.DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, fb=frozen_bits):
            return sc_decode(llr_ch, fb.astype(bool)), None

        print(f"\nSC 仿真: N={N}, K={K}")
        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=run_exp1.EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=run_exp1.MAX_FRAMES,
            min_errors=run_exp1.MIN_ERRORS,
            info_indices=info_idx,
            verbose=True,
        )
        label = f"SC, N={N}, K={K}"
        all_results[label] = results
        save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

    shannon_db = find_capacity_limit(run_exp1.RATE)
    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={run_exp1.RATE})",
        save_path="results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )


def run_exp2_sim():
    run_exp2.run_validations()
    N = run_exp2.N
    K = run_exp2.K
    info_idx, _, _ = ga_construction(N, K, run_exp2.DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)
    all_results = {}

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bool), None

    results_sc = run_simulation(
        N, K, run_exp2.EB_N0_RANGE, sc_decoder, "sc",
        run_exp2.MAX_FRAMES, run_exp2.MIN_ERRORS, info_indices=info_idx, verbose=True,
    )
    all_results["SC (L=1)"] = results_sc
    save_results_csv(results_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in run_exp2.L_LIST:
        def scl_decoder(llr_ch, _L=L):
            u_hat, _ = SCLDecoder(N, frozen_bool, list_size=_L).decode(llr_ch)
            return u_hat, None

        print(f"\nSCL 仿真: L={L}")
        results = run_simulation(
            N, K, run_exp2.EB_N0_RANGE, scl_decoder, "scl",
            run_exp2.MAX_FRAMES, run_exp2.MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        all_results[f"SCL (L={L})"] = results
        save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    def cascl_decoder(llr_ch):
        u_hat, _ = SCLDecoder(
            N, frozen_bool, list_size=8, crc_length=run_exp2.CRC_LENGTH
        ).decode(llr_ch)
        return u_hat, None

    print("\nCA-SCL 仿真")
    results_cascl = run_simulation(
        N, K, run_exp2.EB_N0_RANGE, cascl_decoder, "scl",
        run_exp2.MAX_FRAMES, run_exp2.MIN_ERRORS, crc_length=run_exp2.CRC_LENGTH,
        info_indices=info_idx, verbose=True,
    )
    all_results[f"CA-SCL (L=8, CRC={run_exp2.CRC_LENGTH})"] = results_cascl
    save_results_csv(results_cascl, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

    shannon_db = find_capacity_limit(run_exp2.RATE)
    plot_bler_curves(
        all_results,
        f"SCL vs SC BLER (N={N}, R={run_exp2.RATE})",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )

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


def run_exp3_sim():
    run_exp3.run_validations()
    for N in run_exp3.N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, run_exp3.DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        frozen_bool = frozen_bits.astype(bool)
        all_results = {}

        def sc_d(llr_ch):
            return sc_decode(llr_ch, frozen_bool), None

        print(f"\n实验三 N={N} - SC")
        r_sc = run_simulation(
            N, K, run_exp3.EB_N0_RANGE, sc_d, "sc",
            run_exp3.MAX_FRAMES, run_exp3.MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr_ch):
            u, _ = SCLDecoder(N, frozen_bool, list_size=4).decode(llr_ch)
            return u, None

        print(f"实验三 N={N} - SCL")
        r_scl = run_simulation(
            N, K, run_exp3.EB_N0_RANGE, scl_d, "scl",
            run_exp3.MAX_FRAMES, run_exp3.MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp_decoder = BPDecoder(N, frozen_bool, max_iter=run_exp3.MAX_ITER)

        def bp_d(llr_ch):
            u_hat, num_iters = bp_decoder.decode(llr_ch)
            return u_hat, num_iters

        print(f"实验三 N={N} - BP")
        r_bp = run_simulation(
            N, K, run_exp3.EB_N0_RANGE, bp_d, "bp",
            run_exp3.MAX_FRAMES, run_exp3.MIN_ERRORS, info_indices=info_idx, verbose=True,
        )
        all_results[f"BP (max_iter={run_exp3.MAX_ITER})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        shannon_db = find_capacity_limit(run_exp3.RATE)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={run_exp3.RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_n0_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_n0_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={run_exp3.MAX_ITER})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()


def main():
    os.makedirs("results", exist_ok=True)
    print("=" * 60, "\n实验一：SC 仿真\n", "=" * 60, sep="")
    run_exp1_sim()
    print("\n", "=" * 60, "\n实验二：SCL 仿真\n", "=" * 60, sep="")
    run_exp2_sim()
    print("\n", "=" * 60, "\n实验三：BP 仿真\n", "=" * 60, sep="")
    run_exp3_sim()
    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
