"""快速生成 results/ 下的曲线与 CSV（降低帧数，供自动化/PR 使用）"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import (
    load_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
    find_capacity_limit,
)

DESIGN = 2.5
RATE = 0.5
# SC 在 N>=256 时约需 Eb/N0>=7 dB 才有明显收敛，仿真范围覆盖中高信噪比
EB = np.arange(4.0, 10.5, 0.5)
MAX_F = 3000
MIN_E = 80


def main():
    os.makedirs("results", exist_ok=True)
    save_frozen_set_info([256, 512, 1024], None, DESIGN, "results/frozen_sets.txt")

    # ---------- 实验一：SC ----------
    all_sc = {}
    for N in [256, 512, 1024]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0
        fbool = fb.astype(bool)

        def dec_sc(llr, fb=fbool):
            return sc_decode(llr, fb), None

        r = run_simulation(
            N,
            K,
            EB,
            dec_sc,
            "sc",
            MAX_F,
            MIN_E,
            info_indices=info_idx,
            frozen_bits=fb,
            seed=42 + N,
        )
        path = f"results/exp1_sc_N{N}_R0.5.csv"
        save_results_csv(r, path)
        all_sc[f"SC, N={N}, K={K}"] = r

    plot_bler_curves(
        all_sc,
        f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        "results/fig1_sc_bler.png",
        find_capacity_limit(RATE),
    )

    # ---------- 实验二：SCL @ N=512 ----------
    N, K = 512, 256
    info_idx, _, _ = ga_construction(N, K, DESIGN)
    fb = np.ones(N, dtype=int)
    fb[info_idx] = 0
    fbool = fb.astype(bool)
    all2 = {}

    def dec_sc(llr):
        return sc_decode(llr, fbool), None

    all2["SC (L=1)"] = run_simulation(
        N, K, EB, dec_sc, "sc", MAX_F, MIN_E, info_indices=info_idx, frozen_bits=fb, seed=512
    )
    save_results_csv(all2["SC (L=1)"], f"results/exp2_sc_N{N}_R0.5.csv")

    for L in [2, 4, 8]:
        scl = SCLDecoder(N, fbool, list_size=L)

        def dec_scl(llr, d=scl):
            u, _ = d.decode(llr)
            return u, None

        label = f"SCL (L={L})"
        all2[label] = run_simulation(
            N, K, EB, dec_scl, "scl", MAX_F, MIN_E, info_indices=info_idx, frozen_bits=fb, seed=512 + L
        )
        save_results_csv(all2[label], f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    save_results_csv(all2["SCL (L=4)"], f"results/exp2_scl_N{N}_R0.5.csv")

    cascl = SCLDecoder(N, fbool, list_size=8, crc_length=8)

    def dec_ca(llr):
        u, _ = cascl.decode(llr)
        return u, None

    all2["CA-SCL (L=8, CRC=8)"] = run_simulation(
        N,
        K,
        EB,
        dec_ca,
        "scl",
        MAX_F,
        MIN_E,
        crc_length=8,
        info_indices=info_idx,
        frozen_bits=fb,
        seed=520,
    )
    save_results_csv(all2["CA-SCL (L=8, CRC=8)"], f"results/exp2_cascl_L8_N{N}_R0.5.csv")

    plot_bler_curves(
        all2,
        f"SCL vs SC BLER (N={N}, R={RATE})",
        "results/fig2_scl_bler.png",
        find_capacity_limit(RATE),
    )

    labels = list(all2.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all2.values()]
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

    # ---------- 实验三：SC / SCL / BP ----------
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN)
        fb = np.ones(N, dtype=int)
        fb[info_idx] = 0
        fbool = fb.astype(bool)
        all3 = {}

        def dec_sc(llr):
            return sc_decode(llr, fbool), None

        all3["SC"] = run_simulation(
            N, K, EB, dec_sc, "sc", MAX_F, MIN_E, info_indices=info_idx, frozen_bits=fb, seed=3000 + N
        )
        save_results_csv(all3["SC"], f"results/exp3_sc_N{N}_R0.5.csv")

        scl = SCLDecoder(N, fbool, list_size=4)

        def dec_scl(llr):
            u, _ = scl.decode(llr)
            return u, None

        all3["SCL (L=4)"] = run_simulation(
            N, K, EB, dec_scl, "scl", MAX_F, MIN_E, info_indices=info_idx, frozen_bits=fb, seed=3100 + N
        )
        save_results_csv(all3["SCL (L=4)"], f"results/exp3_scl_N{N}_R0.5.csv")

        bp = BPDecoder(N, fbool, max_iter=50)

        def dec_bp(llr):
            u, it = bp.decode(llr)
            return u, it

        all3["BP (max_iter=50)"] = run_simulation(
            N, K, EB, dec_bp, "bp", MAX_F, MIN_E, info_indices=info_idx, frozen_bits=fb, seed=3200 + N
        )
        save_results_csv(all3["BP (max_iter=50)"], f"results/exp3_bp_N{N}_R0.5.csv")

        plot_bler_curves(
            all3,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            find_capacity_limit(RATE),
        )

        eb_vals = [r["eb_n0_db"] for r in all3["BP (max_iter=50)"]]
        iters = [r["avg_iters"] for r in all3["BP (max_iter=50)"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_vals, iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter=50)")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()

    print("quick all done")


if __name__ == "__main__":
    from run_exp1 import _unit_tests

    _unit_tests()
    main()
