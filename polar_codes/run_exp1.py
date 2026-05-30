"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from simulation import run_simulation
from utils import (
    save_results_csv,
    plot_bler_curves,
    save_frozen_set_info,
    find_capacity_limit,
)

os.makedirs("results", exist_ok=True)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=bool)
frozen_t[info_t] = False
rng_t = np.random.default_rng(0)
err_sc = 0
for _ in range(100):
    ut = np.zeros(N_t, dtype=int)
    ut[info_t] = rng_t.integers(0, 2, K_t)
    llr = compute_llr(bpsk_modulate(polar_encode(ut)), 0.01)
    if not np.array_equal(sc_decode(llr, frozen_t), ut):
        err_sc += 1
assert err_sc == 0, f"SC 无损验证失败: {err_sc}/100"

scl1 = SCLDecoder(N_t, frozen_t, list_size=1)
ut = np.zeros(N_t, dtype=int)
ut[info_t] = rng_t.integers(0, 2, K_t)
llr = compute_llr(bpsk_modulate(polar_encode(ut)), 0.01)
uh_scl, _ = scl1.decode(llr)
assert np.array_equal(uh_scl, sc_decode_recursive(llr, frozen_t)), "L=1 SCL 应等价 SC"
print("单元测试通过。")

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = int(os.environ.get("POLAR_MAX_FRAMES", "100000"))
MIN_ERRORS = int(os.environ.get("POLAR_MIN_ERRORS", "100"))
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}
for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")

    info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

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
