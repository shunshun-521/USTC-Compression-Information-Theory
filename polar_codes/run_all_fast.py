"""
快速运行全部实验（降低帧数以在合理时间内完成）
完整参数请直接运行 run_exp1/2/3.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

MAX_FRAMES = 8000
MIN_ERRORS = 40
DESIGN_EBN0 = 2.5
RATE = 0.5

os.makedirs("results", exist_ok=True)


def run_exp1():
    N_LIST = [256, 512, 1024]
    EB_N0_RANGE = np.arange(0.0, 5.5, 0.5)
    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")
    all_results = {}
    for N in N_LIST:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0

        def dec(llr):
            return sc_decode(llr, fb), None

        print(f"exp1 SC N={N}")
        r = run_simulation(N, K, EB_N0_RANGE, dec, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all_results[f"SC, N={N}, K={K}"] = r
        save_results_csv(r, f"results/exp1_sc_N{N}_R0.5.csv")
    plot_bler_curves(all_results, f"SC BLER (R={RATE})", "results/fig1_sc_bler.png", find_capacity_limit(RATE))


def run_exp2():
    N = 512
    K = N // 2
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)
    CRC_LENGTH = 8
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    fb = np.ones(N, dtype=int)
    fb[info_idx] = 0
    all_results = {}

    def sc_dec(llr):
        return sc_decode(llr, fb), None

    print("exp2 SC")
    r = run_simulation(N, K, EB_N0_RANGE, sc_dec, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
    all_results["SC (L=1)"] = r
    save_results_csv(r, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in [2, 4, 8]:
        print(f"exp2 SCL L={L}")

        def scl_dec(llr, _L=L):
            u, _ = SCLDecoder(N, fb, list_size=_L).decode(llr)
            return u, None

        r = run_simulation(N, K, EB_N0_RANGE, scl_dec, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all_results[f"SCL (L={L})"] = r
        save_results_csv(r, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    print("exp2 CA-SCL")

    def cascl_dec(llr):
        u, _ = SCLDecoder(N, fb, list_size=8, crc_length=CRC_LENGTH).decode(llr)
        return u, None

    r = run_simulation(
        N, K, EB_N0_RANGE, cascl_dec, "scl", MAX_FRAMES, MIN_ERRORS,
        crc_length=CRC_LENGTH, info_indices=info_idx,
    )
    all_results[f"CA-SCL (L=8, CRC={CRC_LENGTH})"] = r
    save_results_csv(r, f"results/exp2_cascl_L8_N{N}_R0.5.csv")
    save_results_csv(r, f"results/exp2_scl_N{N}_R0.5.csv")

    plot_bler_curves(all_results, f"SCL vs SC (N={N})", "results/fig2_scl_bler.png", find_capacity_limit(RATE))

    labels = list(all_results.keys())
    avg_times = [np.mean([x["avg_decode_time"] for x in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()


def run_exp3():
    EB_N0_RANGE = np.arange(1.0, 5.5, 0.5)
    MAX_ITER = 50
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0
        all_results = {}

        def sc_d(llr):
            return sc_decode(llr, fb), None

        print(f"exp3 SC N={N}")
        r_sc = run_simulation(N, K, EB_N0_RANGE, sc_d, "sc", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all_results["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_d(llr):
            u, _ = SCLDecoder(N, fb, list_size=4).decode(llr)
            return u, None

        print(f"exp3 SCL N={N}")
        r_scl = run_simulation(N, K, EB_N0_RANGE, scl_d, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all_results["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp = BPDecoder(N, fb, max_iter=MAX_ITER)

        def bp_d(llr):
            u, it = bp.decode(llr)
            return u, it

        print(f"exp3 BP N={N}")
        r_bp = run_simulation(N, K, EB_N0_RANGE, bp_d, "bp", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx)
        all_results[f"BP (max_iter={MAX_ITER})"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        plot_bler_curves(all_results, f"SC vs SCL vs BP (N={N})", f"results/fig3_bp_N{N}_bler.png", find_capacity_limit(RATE))

        eb = [x["eb_n0_db"] for x in r_bp]
        iters = [x["avg_iters"] for x in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb, iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Iterations (N={N})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()


if __name__ == "__main__":
    run_exp1()
    run_exp2()
    run_exp3()
    print("全部实验完成。")
