#!/usr/bin/env python3
"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import find_capacity_limit, plot_bler_curves, save_frozen_set_info, save_results_csv

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
from encoder import polar_encode

x = polar_encode(u)
from encoder import polar_generator_matrix

assert np.array_equal(x, (u @ polar_generator_matrix(4)) % 2), f"编码器错误: {x}"

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=bool)
frozen_t[info_t] = False
rng_t = np.random.default_rng(0)
sigma_t = eb_n0_to_sigma(10.0, 0.5)
for _ in range(100):
    payload = rng_t.integers(0, 2, size=K_t)
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = payload
    llr_t = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), sigma_t, rng_t), sigma_t)
    assert np.array_equal(sc_decode(llr_t, frozen_t)[info_t], payload)
print("单元测试通过。")

# ========== 参数设置 ==========
QUICK = os.environ.get("POLAR_QUICK", "0") == "1"
N_LIST = [64, 128] if QUICK else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if QUICK else 100000
MIN_ERRORS = 10 if QUICK else 100
EB_N0_RANGE = np.arange(1.0, 3.5, 0.5) if QUICK else np.arange(0.0, 5.5, 0.25)

os.makedirs("results", exist_ok=True)
save_frozen_set_info(N_LIST if not QUICK else [256, 512], None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}
for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

    results = run_simulation(
        N=N, K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
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
