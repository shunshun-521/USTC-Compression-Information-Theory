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
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)

os.makedirs("results", exist_ok=True)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
from encoder import polar_encode

x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

N_test, K_test = 64, 32
info_idx_t, _, _ = ga_construction(N_test, K_test, 2.5)
frozen_t = np.ones(N_test, dtype=int)
frozen_t[info_idx_t] = 0
from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma

rng = np.random.default_rng(0)
sigma_t = eb_n0_to_sigma(10.0, 0.5)
for _ in range(100):
    info = rng.integers(0, 2, K_test)
    u_t = np.zeros(N_test, dtype=int)
    u_t[info_idx_t] = info
    x_t = polar_encode(u_t)
    y_t = awgn_channel(bpsk_modulate(x_t), sigma_t, rng=rng)
    llr_t = compute_llr(y_t, sigma_t)
    assert np.array_equal(sc_decode(llr_t, frozen_t)[info_idx_t], info)
print("单元测试通过。")

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

if os.environ.get("POLAR_QUICK") == "1":
    N_LIST = [256, 512]
    EB_N0_RANGE = np.arange(1.0, 4.5, 0.5)
    MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "3000"))
    MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "20"))

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, frozen_idx, llr_means = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, _fb=frozen_bits.copy()):
        return sc_decode(llr_ch, _fb), None

    results = run_simulation(
        N=N,
        K=K,
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
