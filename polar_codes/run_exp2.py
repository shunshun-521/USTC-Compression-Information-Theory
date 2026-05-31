"""
实验二：SCL 译码及 CRC 辅助

环境变量 EXP2_STAGE 可选：
  sc | scl_L2 | scl_L4 | scl_L8 | scl_L16 | cascl | plots | all（默认 all）
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_equals_sc
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, load_results_csv, find_capacity_limit

os.makedirs("results", exist_ok=True)

N = 512
RATE = 0.5
K = N // 2
DESIGN_EBN0 = 2.5
CRC_LENGTH = 8


def run_unit_tests():
    from decoder_scl import crc_encode, crc_check

    info = np.ones(10, dtype=int)
    assert crc_check(crc_encode(info, 8), 8), "CRC encode/check 不一致"

    n = 64
    frozen_bits = np.ones(n, dtype=bool)
    frozen_bits[: n // 2] = False
    assert scl_equals_sc(n, frozen_bits), "L=1 时 SCL 应与 SC 等价"
    print("单元测试通过。")


def _sim_params():
    max_frames = int(os.environ.get("POLAR_MAX_FRAMES", 100000))
    min_errors = int(os.environ.get("POLAR_MIN_ERRORS", 100))
    eb_step = float(os.environ.get("POLAR_EB_STEP", "0.25"))
    eb_range = np.arange(1.0, 5.5, eb_step)
    if os.environ.get("POLAR_QUICK"):
        eb_range = np.arange(2.0, 5.5, 0.5)
    return max_frames, min_errors, eb_range


def _frozen_setup():
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    return info_idx, frozen_bits


def run_sc(info_idx, frozen_bits, max_frames, min_errors, eb_range):
    print("SC 基线 (L=1)")

    def sc_decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N, K, eb_range, sc_decoder, "sc", max_frames, min_errors,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, f"results/exp2_sc_N{N}_R0.5.csv")
    return results


def run_scl(L, info_idx, frozen_bits, max_frames, min_errors, eb_range):
    print(f"\nSCL 仿真: N={N}, K={K}, L={L}")
    scl = SCLDecoder(N, frozen_bits, list_size=L, crc_length=0)

    def scl_decoder(llr_ch, _scl=scl):
        u_hat, _ = _scl.decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, eb_range, scl_decoder, "scl", max_frames, min_errors,
        info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, f"results/exp2_scl_L{L}_N{N}_R0.5.csv")
    if L == 8:
        save_results_csv(results, f"results/exp2_scl_N{N}_R0.5.csv")
    return results


def run_cascl(info_idx, frozen_bits, max_frames, min_errors, eb_range):
    print(f"\nCA-SCL 仿真: N={N}, K={K}, L=8, CRC={CRC_LENGTH}")
    cascl = SCLDecoder(N, frozen_bits, list_size=8, crc_length=CRC_LENGTH)

    def cascl_decoder(llr_ch):
        u_hat, _ = cascl.decode(llr_ch)
        return u_hat, None

    results = run_simulation(
        N, K, eb_range, cascl_decoder, "scl", max_frames, min_errors,
        crc_length=CRC_LENGTH, info_indices=info_idx, verbose=True,
    )
    save_results_csv(results, f"results/exp2_cascl_L8_N{N}_R0.5.csv")
    return results


def run_plots():
    mapping = {
        "SC (L=1)": f"results/exp2_sc_N{N}_R0.5.csv",
        "SCL (L=2)": f"results/exp2_scl_L2_N{N}_R0.5.csv",
        "SCL (L=4)": f"results/exp2_scl_L4_N{N}_R0.5.csv",
        "SCL (L=8)": f"results/exp2_scl_L8_N{N}_R0.5.csv",
        "CA-SCL (L=8, CRC=8)": f"results/exp2_cascl_L8_N{N}_R0.5.csv",
    }
    all_results = {}
    for label, path in mapping.items():
        if os.path.exists(path):
            all_results[label] = load_results_csv(path)
    if not all_results:
        print("无实验二 CSV，跳过绘图。")
        return

    shannon_db = find_capacity_limit(RATE)
    plot_bler_curves(
        all_results,
        f"SCL vs SC BLER (N={N}, R={RATE})",
        "results/fig2_scl_bler.png",
        shannon_limit_db=shannon_db,
    )

    labels = list(all_results.keys())
    avg_times = [
        np.mean([r["avg_decode_time"] for r in v]) * 1000 for v in all_results.values()
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
    print("fig2 已更新。")


STAGE_MAP = {
    "sc": ("sc", None),
    "scl_l2": ("scl", 2),
    "scl_l4": ("scl", 4),
    "scl_l8": ("scl", 8),
    "scl_l16": ("scl", 16),
    "cascl": ("cascl", None),
    "plots": ("plots", None),
}


def main():
    run_unit_tests()
    stage = os.environ.get("EXP2_STAGE", "all").strip().lower()
    max_frames, min_errors, eb_range = _sim_params()
    info_idx, frozen_bits = _frozen_setup()

    if stage == "all":
        stages = ["sc", "scl_l2", "scl_l4", "scl_l8", "cascl", "plots"]
    else:
        stages = [stage]

    for s in stages:
        if s not in STAGE_MAP:
            raise ValueError(f"未知 EXP2_STAGE={s}")
        kind, arg = STAGE_MAP[s]
        if kind == "sc":
            run_sc(info_idx, frozen_bits, max_frames, min_errors, eb_range)
        elif kind == "scl":
            run_scl(arg, info_idx, frozen_bits, max_frames, min_errors, eb_range)
        elif kind == "cascl":
            run_cascl(info_idx, frozen_bits, max_frames, min_errors, eb_range)
        elif kind == "plots":
            run_plots()

    if stage == "all":
        print("\n实验二完成。")


if __name__ == "__main__":
    main()
