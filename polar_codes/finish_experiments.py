"""补全未完成的实验二、实验三输出"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from decoder_bp import BPDecoder
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs("results", exist_ok=True)
os.environ.setdefault("POLAR_QUICK", "1")
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", 4000))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", 40))
EB_N0_RANGE = np.arange(2.0, 5.5, 0.5)


def finish_exp2():
    N = 512
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    all_results = {}

    if os.path.exists("results/exp2_sc_N512_R0.5.csv"):
        all_results["SC (L=1)"] = load_results_csv("results/exp2_sc_N512_R0.5.csv")
    for L in [2, 4, 8]:
        path = f"results/exp2_scl_L{L}_N512_R0.5.csv"
        if os.path.exists(path):
            all_results[f"SCL (L={L})"] = load_results_csv(path)

    missing_L = [L for L in [8] if f"SCL (L={L})" not in all_results]
    for L in missing_L:
        print(f"补跑 SCL L={L}")
        scl = SCLDecoder(N, frozen_bits, list_size=L)

        def dec(llr, _s=scl):
            u, _ = _s.decode(llr)
            return u, None

        r = run_simulation(
            N, K, EB_N0_RANGE, dec, "scl", MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
        )
        all_results[f"SCL (L={L})"] = r
        save_results_csv(r, f"results/exp2_scl_L{L}_N512_R0.5.csv")

    if "CA-SCL (L=8, CRC=8)" not in all_results:
        print("补跑 CA-SCL")
        cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=8)

        def dec_c(llr):
            u, _ = cascl.decode(llr)
            return u, None

        r = run_simulation(
            N,
            K,
            EB_N0_RANGE,
            dec_c,
            "scl",
            MAX_FRAMES,
            MIN_ERRORS,
            crc_length=8,
            info_indices=info_idx,
        )
        all_results["CA-SCL (L=8, CRC=8)"] = r
        save_results_csv(r, f"results/exp2_cascl_L8_N512_R0.5.csv")

    if "SCL (L=4)" in all_results:
        save_results_csv(all_results["SCL (L=4)"], "results/exp2_scl_N512_R0.5.csv")

    shannon_db = find_capacity_limit(0.5)
    plot_bler_curves(
        all_results,
        "SCL vs SC BLER (N=512, R=0.5)",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )
    labels = list(all_results.keys())
    avg_times = [np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()]
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
    print("实验二输出已补全。")


def finish_exp3():
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        all_results = {}

        for name, dtype, factory in [
            ("SC", "sc", lambda: sc_decode),
            ("SCL (L=4)", "scl", lambda: SCLDecoder(N, frozen_bits, 4)),
            ("BP (max_iter=50)", "bp", lambda: BPDecoder(N, frozen_bits, 50)),
        ]:
            path = f"results/exp3_{name.split()[0].lower()}_N{N}_R0.5.csv".replace(
                "(l=4)", "scl"
            )
            if name == "SC":
                path = f"results/exp3_sc_N{N}_R0.5.csv"
            elif "SCL" in name:
                path = f"results/exp3_scl_N{N}_R0.5.csv"
            else:
                path = f"results/exp3_bp_N{N}_R0.5.csv"

            if os.path.exists(path):
                all_results[name] = load_results_csv(path)
                continue

            print(f"补跑 N={N} {name}")
            obj = factory()

            if dtype == "sc":

                def dec(llr, _f=frozen_bits):
                    return sc_decode(llr, _f), None
            elif dtype == "scl":

                def dec(llr, _s=obj):
                    u, _ = _s.decode(llr)
                    return u, None
            else:

                def dec(llr, _b=obj):
                    u, it = _b.decode(llr)
                    return u, it

            r = run_simulation(
                N, K, EB_N0_RANGE, dec, dtype, MAX_FRAMES, MIN_ERRORS, info_indices=info_idx
            )
            all_results[name] = r
            save_results_csv(r, path)

        shannon_db = find_capacity_limit(0.5)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R=0.5)",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )
        r_bp = all_results.get("BP (max_iter=50)")
        if r_bp:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([x["eb_n0_db"] for x in r_bp], [x["avg_iters"] for x in r_bp], "o-", color="purple")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("Avg Iterations")
            ax.set_title(f"BP Average Iterations (N={N})")
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
            plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
            plt.close()
    print("实验三输出已补全。")


if __name__ == "__main__":
    finish_exp2()
    finish_exp3()
