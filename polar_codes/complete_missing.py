"""补全缺失的实验结果文件。"""
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
from utils import find_capacity_limit, plot_bler_curves, save_results_csv

EB = np.array([4.0, 5.0])
MAX_F = 10
MIN_E = 3


def run_exp2_remaining():
    N, K = 512, 256
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bool = np.ones(N, dtype=bool)
    frozen_bool[info_idx] = False
    all_results = {}

    for L in [8]:
        def scl_d(llr, _L=L):
            u, _ = SCLDecoder(N, frozen_bool, list_size=_L).decode(llr)
            return u, None

        print(f"SCL L={L}")
        r = run_simulation(N, K, EB, scl_d, "scl", MAX_F, MIN_E, info_indices=info_idx)
        save_results_csv(r, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")
        all_results[f"SCL (L={L})"] = r

    def cascl_d(llr):
        u, _ = SCLDecoder(N, frozen_bool, list_size=8, crc_length=8).decode(llr)
        return u, None

    print("CA-SCL")
    r = run_simulation(N, K, EB, cascl_d, "scl", MAX_F, MIN_E, crc_length=8, info_indices=info_idx)
    save_results_csv(r, f"results/exp2_cascl_L8_N{N}_R0.5.csv")
    all_results["CA-SCL (L=8, CRC=8)"] = r

    # 合并已有 SC/SCL 结果绘图
    from utils import load_results_csv
    all_results["SC (L=1)"] = load_results_csv("results/exp2_sc_N512_R0.5.csv")
    all_results["SCL (L=2)"] = load_results_csv("results/exp2_scl_L2_N512_R0.5.csv")
    all_results["SCL (L=4)"] = load_results_csv("results/exp2_scl_L4_N512_R0.5.csv")

    plot_bler_curves(all_results, "SCL vs SC BLER (N=512, R=0.5)",
                     "results/fig2_scl_bler.png", find_capacity_limit(0.5))

    labels = list(all_results.keys())
    avg_times = [np.mean([x["avg_decode_time"] for x in v]) * 1000 for v in all_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, avg_times)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Avg Decode Time (ms)")
    ax.set_title("Decoding Time vs List Size (N=512)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("results/fig2_decode_time.png", dpi=150)
    plt.savefig("results/fig2_decode_time.pdf")
    plt.close()


def run_exp3_all():
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bool = np.ones(N, dtype=bool)
        frozen_bool[info_idx] = False
        all_results = {}

        def sc_d(llr):
            return sc_decode(llr, frozen_bool), None

        print(f"N={N} SC")
        r_sc = run_simulation(N, K, EB, sc_d, "sc", MAX_F * 2, MIN_E, info_indices=info_idx)
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")
        all_results["SC"] = r_sc

        def scl_d(llr):
            u, _ = SCLDecoder(N, frozen_bool, list_size=4).decode(llr)
            return u, None

        print(f"N={N} SCL")
        r_scl = run_simulation(N, K, EB, scl_d, "scl", MAX_F, MIN_E, info_indices=info_idx)
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")
        all_results["SCL (L=4)"] = r_scl

        bp = BPDecoder(N, frozen_bool, max_iter=50)

        def bp_d(llr):
            u, it = bp.decode(llr)
            return u, it

        print(f"N={N} BP")
        r_bp = run_simulation(N, K, EB, bp_d, "bp", MAX_F * 2, MIN_E, info_indices=info_idx)
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")
        all_results["BP (max_iter=50)"] = r_bp

        plot_bler_curves(all_results, f"SC vs SCL vs BP (N={N}, R=0.5)",
                         f"results/fig3_bp_N{N}_bler.png", find_capacity_limit(0.5))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([x["eb_n0_db"] for x in r_bp], [x["avg_iters"] for x in r_bp], "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_exp2_remaining()
    run_exp3_all()
    print("ALL DONE")
