"""快速生成实验结果（缩减仿真规模，用于自动化验证）"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from run_exp1 import run_unit_tests
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

os.makedirs("results", exist_ok=True)

MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "20"))
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "300"))
EB_N0 = np.array([2.0, 3.0, 4.0, 5.0])
DESIGN_EBN0 = 2.5
RATE = 0.5


def main():
    run_unit_tests()
    save_frozen_set_info([256, 512, 1024], None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_sc = {}
    for N in [256, 512]:
        K = N // 2
        info, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen = np.ones(N, dtype=int)
        frozen[info] = 0
        frozen_bool = frozen.astype(bool)

        def decoder(llr):
            return sc_decode(llr, frozen), None

        results = run_simulation(
            N, K, EB_N0, decoder, "sc", MAX_FRAMES, MIN_ERRORS,
            info_indices=info, frozen_bits=frozen_bool, verbose=True,
        )
        all_sc[f"SC, N={N}, K={K}"] = results
        save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

    plot_bler_curves(all_sc, "SC Decoder BLER vs Eb/N0 (R=0.5)",
                     "results/fig1_sc_bler.png", find_capacity_limit(RATE))

    N, K = 512, 256
    info, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    frozen_bool = frozen.astype(bool)
    all_scl = {}

    def sc_d(llr):
        return sc_decode(llr, frozen), None

    r_sc = run_simulation(N, K, EB_N0, sc_d, "sc", MAX_FRAMES, MIN_ERRORS,
                          info_indices=info, frozen_bits=frozen_bool, verbose=True)
    all_scl["SC (L=1)"] = r_sc
    save_results_csv(r_sc, f"results/exp2_sc_N{N}_R0.5.csv")

    for L in [2, 4, 8]:
        def scl_d(llr, _L=L):
            u, _ = SCLDecoder(N, frozen, list_size=_L).decode(llr)
            return u, None

        r = run_simulation(N, K, EB_N0, scl_d, "scl", MAX_FRAMES, MIN_ERRORS,
                           info_indices=info, frozen_bits=frozen_bool, verbose=True)
        all_scl[f"SCL (L={L})"] = r
        save_results_csv(r, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")

    def cascl_d(llr):
        u, _ = SCLDecoder(N, frozen, list_size=8, crc_length=8).decode(llr)
        return u, None

    r_ca = run_simulation(N, K, EB_N0, cascl_d, "scl", MAX_FRAMES, MIN_ERRORS,
                          crc_length=8, info_indices=info, frozen_bits=frozen_bool, verbose=True)
    all_scl["CA-SCL (L=8, CRC=8)"] = r_ca
    save_results_csv(r_ca, f"results/exp2_cascl_L8_N{N}_R0.5.csv")

    plot_bler_curves(all_scl, f"SCL vs SC BLER (N={N}, R={RATE})",
                     "results/fig2_scl_bler.png", find_capacity_limit(RATE))

    labels = list(all_scl.keys())
    avg_times = [np.mean([x["avg_decode_time"] for x in v]) * 1000 for v in all_scl.values()]
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

    for N in [256, 512]:
        K = N // 2
        info, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen = np.ones(N, dtype=int)
        frozen[info] = 0
        frozen_bool = frozen.astype(bool)
        all_bp = {}

        def sc_dec(llr):
            return sc_decode(llr, frozen), None

        r_sc = run_simulation(N, K, EB_N0, sc_dec, "sc", MAX_FRAMES, MIN_ERRORS,
                              info_indices=info, frozen_bits=frozen_bool, verbose=True)
        all_bp["SC"] = r_sc
        save_results_csv(r_sc, f"results/exp3_sc_N{N}_R0.5.csv")

        def scl_dec(llr):
            u, _ = SCLDecoder(N, frozen, list_size=4).decode(llr)
            return u, None

        r_scl = run_simulation(N, K, EB_N0, scl_dec, "scl", MAX_FRAMES, MIN_ERRORS,
                               info_indices=info, frozen_bits=frozen_bool, verbose=True)
        all_bp["SCL (L=4)"] = r_scl
        save_results_csv(r_scl, f"results/exp3_scl_N{N}_R0.5.csv")

        bp = BPDecoder(N, frozen, max_iter=50)

        def bp_dec(llr):
            u, it = bp.decode(llr)
            return u, it

        r_bp = run_simulation(N, K, EB_N0, bp_dec, "bp", MAX_FRAMES, MIN_ERRORS,
                              info_indices=info, frozen_bits=frozen_bool, verbose=True)
        all_bp["BP (max_iter=50)"] = r_bp
        save_results_csv(r_bp, f"results/exp3_bp_N{N}_R0.5.csv")

        plot_bler_curves(all_bp, f"SC vs SCL vs BP (N={N}, R={RATE})",
                         f"results/fig3_bp_N{N}_bler.png", find_capacity_limit(RATE))

        eb_vals = [x["eb_n0_db"] for x in r_bp]
        iters = [x["avg_iters"] for x in r_bp]
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

    print("快速实验完成。")


if __name__ == "__main__":
    main()
