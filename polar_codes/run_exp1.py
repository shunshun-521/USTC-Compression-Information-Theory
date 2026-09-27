"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from decoder_sc import sc_decode, sc_decode_recursive
from simulation import run_simulation
from utils import (
    save_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    find_capacity_limit,
)
from channel import eb_n0_to_sigma, bpsk_modulate, awgn_channel, compute_llr

os.makedirs("results", exist_ok=True)

# ========== 单元测试 ==========
assert np.array_equal(polar_encode([1, 1, 1, 1]), [0, 0, 0, 1]), "编码器错误"
assert np.array_equal(polar_encode([1, 0, 1, 1]), [1, 1, 0, 1]), "编码器错误"


def _sc_lossless_test():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen_bits.astype(bool))
        if not np.array_equal(uh[info_idx], u[info_idx]):
            raise AssertionError("SC 无损校验失败")


_sc_lossless_test()
print("单元测试通过：编码器、SC 无损校验")

# ========== GA 构造验证 ==========
info8, frozen8, _ = ga_construction(8, 4, 2.5)
print("N=8 info:", info8, "frozen:", frozen8)
info256, _, _ = ga_construction(256, 128, 2.5)
print("N=256 info (first 20):", info256[:20])

# ========== 参数设置 ==========
QUICK = os.environ.get("POLAR_QUICK", "0") == "1"
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if QUICK else 100000
MIN_ERRORS = 10 if QUICK else 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.5 if QUICK else 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, fb=frozen_bits.astype(bool)):
        return sc_decode(llr_ch, fb), None

    results = run_simulation(
        N=N,
        K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits.astype(bool),
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
