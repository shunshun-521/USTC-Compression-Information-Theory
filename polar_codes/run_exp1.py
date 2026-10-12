"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=int)
frozen_t[info_t] = 0
rng = np.random.default_rng(0)
ok = 0
for _ in range(100):
    u_t = np.zeros(N_t, dtype=int)
    u_t[info_t] = rng.integers(0, 2, K_t)
    x_t = polar_encode(u_t)
    llr_t = compute_llr(bpsk_modulate(x_t), eb_n0_to_sigma(10.0, 0.5))
    if np.array_equal(sc_decode(llr_t, frozen_t), u_t):
        ok += 1
print(f"SC 无损抽检: {ok}/100 帧正确（N={N_t}，供调试参考）")
if ok < 50:
    print("警告: SC 抽检通过率偏低，请检查译码器与 LLR 约定。")

os.makedirs("results", exist_ok=True)

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 5000
MIN_ERRORS = 30
EB_N0_RANGE = np.arange(0.0, 5.5, 0.5)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt", rate=RATE)

all_results = {}
for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}\nSC 仿真: N={N}, K={K}, R={RATE}\n{'=' * 60}")
    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch):
        return sc_decode(llr_ch, frozen_bits), None

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
plot_bler_curves(
    all_results,
    title=f"SC Decoder BLER vs Eb/N0 (R={RATE})",
    save_path="results/fig1_sc_bler.png",
    shannon_limit_db=shannon_db,
)
print("\n实验一完成。")
