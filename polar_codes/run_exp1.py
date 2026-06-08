"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, verify_sc_decoders
from simulation import run_simulation
from utils import (
    save_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    find_capacity_limit,
)


def run_unit_tests():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_br = polar_encode(np.array([1, 0, 1, 1]))
    assert len(x) == 4, "编码器输出长度错误"
    from construction import ga_construction as gc

    info, frozen, _ = gc(8, 4, 2.5)
    print("N=8,K=4 info:", info, "frozen:", frozen)
    info256, _, _ = gc(256, 128, 2.5)
    print("N=256 first 20 info:", info256[:20])
    verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0)
    print("单元测试通过。")


os.makedirs("results", exist_ok=True)

QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

N_LIST = [256, 512] if QUICK else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 5000 if QUICK else 100000
MIN_ERRORS = 20 if QUICK else 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.5 if QUICK else 0.25)

if __name__ == "__main__":
    run_unit_tests()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")

        info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch, fb=frozen_bits):
            return sc_decode(llr_ch, fb), None

        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            info_idx=info_idx,
            frozen_bits=frozen_bits,
            verbose=True,
        )

        label = f"SC, N={N}, K={K}"
        all_results[label] = results
        save_results_csv(results, f"results/exp1_sc_N{N}_R0.5.csv")

    shannon_db = find_capacity_limit(RATE)
    print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")
    plot_bler_curves(
        all_results,
        title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
        save_path="results/fig1_sc_bler.png",
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")
