"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)

os.makedirs("results", exist_ok=True)

_QUICK = os.environ.get("POLAR_QUICK", "0") == "1"

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

from channel import awgn_channel, bpsk_modulate, compute_llr

N_chk, K_chk = 64, 32
info_chk, _, _ = ga_construction(N_chk, K_chk, 2.5)
frozen_chk = np.ones(N_chk, dtype=int)
frozen_chk[info_chk] = 0
sigma_chk = eb_n0_to_sigma(10.0, K_chk / N_chk)
rng_chk = np.random.default_rng(123)
err_sc = 0
for _ in range(100):
    u_t = np.zeros(N_chk, dtype=int)
    payload = rng_chk.integers(0, 2, size=K_chk)
    u_t[info_chk] = payload
    llr_t = compute_llr(
        awgn_channel(bpsk_modulate(polar_encode(u_t)), sigma_chk, rng_chk), sigma_chk
    )
    if not np.array_equal(sc_decode(llr_t, frozen_chk), u_t):
        err_sc += 1
assert err_sc == 0, f"SC 无损校验失败: {err_sc}/100"

from decoder_scl import SCLDecoder

u_t = np.zeros(N_chk, dtype=int)
u_t[info_chk] = rng_chk.integers(0, 2, size=K_chk)
llr_t = compute_llr(bpsk_modulate(polar_encode(u_t)), 0.01)
uh_sc = sc_decode(llr_t, frozen_chk)
uh_scl, _ = SCLDecoder(N_chk, frozen_chk, list_size=1).decode(llr_t)
assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应与 SC 等价"

# ========== 参数设置 ==========
N_LIST = [256, 512] if _QUICK else [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if _QUICK else 100000
MIN_ERRORS = 20 if _QUICK else 100
EB_N0_RANGE = (
    np.arange(1.0, 3.5, 0.5) if _QUICK else np.arange(0.0, 5.5, 0.25)
)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

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
