"""
补全实验三：BP 仿真（含增量保存）
对 BP 使用较低帧数上限以控制运行时间。
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
from utils import save_results_csv, plot_bler_curves, find_capacity_limit, load_results_csv

os.makedirs("results", exist_ok=True)

RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_ITER = 50
EB_N0_RANGE = np.arange(1.0, 5.5, 0.25)

# SC/SCL 使用标准参数；BP 译码较慢，降低帧数上限
SC_MAX_FRAMES = 100000
SC_MIN_ERRORS = 100
BP_MAX_FRAMES = 5000
BP_MIN_ERRORS = 20


def run_bp_with_checkpoint(N, info_idx, frozen_bits, csv_path):
    """逐信噪比点仿真并增量写入 CSV。"""
    K = N // 2
    bp = BPDecoder(N, frozen_bits, max_iter=MAX_ITER)
    results = load_results_csv(csv_path) if os.path.exists(csv_path) else []
    done_eb = {r["eb_n0_db"] for r in results}

    for eb_n0_db in EB_N0_RANGE:
        if eb_n0_db in done_eb:
            print(f"  跳过已完成 Eb/N0={eb_n0_db:.2f}dB")
            continue

        print(f"  BP N={N}, Eb/N0={eb_n0_db:.2f}dB ...")
        point_results = run_simulation(
            N,
            K,
            [eb_n0_db],
            lambda llr, _bp=bp: _bp.decode(llr),
            "bp",
            BP_MAX_FRAMES,
            BP_MIN_ERRORS,
            info_indices=info_idx,
            verbose=True,
        )
        results.append(point_results[0])
        save_results_csv(results, csv_path)
        print(f"  已保存 -> {csv_path}")

    return results


def main():
    for N in [256, 512]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        sc_csv = f"results/exp3_sc_N{N}_R0.5.csv"
        scl_csv = f"results/exp3_scl_N{N}_R0.5.csv"
        bp_csv = f"results/exp3_bp_N{N}_R0.5.csv"

        if not os.path.exists(sc_csv):
            print(f"\nSC 仿真 N={N} ...")

            def sc_d(llr):
                return sc_decode(llr, frozen_bits), None

            r_sc = run_simulation(
                N, K, EB_N0_RANGE, sc_d, "sc",
                SC_MAX_FRAMES, SC_MIN_ERRORS, info_indices=info_idx, verbose=True,
            )
            save_results_csv(r_sc, sc_csv)
        else:
            print(f"\n加载已有 SC 结果: {sc_csv}")
            r_sc = load_results_csv(sc_csv)

        if not os.path.exists(scl_csv):
            print(f"\nSCL 仿真 N={N} ...")

            def scl_d(llr):
                u, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
                return u, None

            r_scl = run_simulation(
                N, K, EB_N0_RANGE, scl_d, "scl",
                SC_MAX_FRAMES, SC_MIN_ERRORS, info_indices=info_idx, verbose=True,
            )
            save_results_csv(r_scl, scl_csv)
        else:
            print(f"\n加载已有 SCL 结果: {scl_csv}")
            r_scl = load_results_csv(scl_csv)

        print(f"\nBP 仿真 N={N} ...")
        r_bp = run_bp_with_checkpoint(N, info_idx, frozen_bits, bp_csv)

        all_results = {
            "SC": r_sc,
            "SCL (L=4)": r_scl,
            f"BP (max_iter={MAX_ITER})": r_bp,
        }

        shannon_db = find_capacity_limit(RATE)
        plot_bler_curves(
            all_results,
            f"SC vs SCL vs BP (N={N}, R={RATE})",
            f"results/fig3_bp_N{N}_bler.png",
            shannon_limit_db=shannon_db,
        )

        eb_vals = [r["eb_n0_db"] for r in r_bp]
        avg_iters = [r["avg_iters"] for r in r_bp]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(eb_vals, avg_iters, "o-", color="purple")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg Iterations")
        ax.set_title(f"BP Average Iterations (N={N}, max_iter={MAX_ITER})")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/fig3_bp_N{N}_iters.png", dpi=150)
        plt.savefig(f"results/fig3_bp_N{N}_iters.pdf")
        plt.close()
        print(f"N={N} 完成。")

    print("\n实验三全部完成。")


if __name__ == "__main__":
    main()
