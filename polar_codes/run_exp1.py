"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs("results", exist_ok=True)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

N_test, K_test = 64, 32
info_t, _, _ = ga_construction(N_test, K_test, 2.5)
frozen_t = np.ones(N_test, dtype=bool)
frozen_t[info_t] = False
rng = np.random.default_rng(0)
err_sc = 0
for _ in range(100):
    u0 = np.zeros(N_test, dtype=np.int8)
    u0[info_t] = rng.integers(0, 2, K_test)
    x0 = polar_encode(u0)
    sigma0 = eb_n0_to_sigma(10.0, K_test / N_test)
    y0 = bpsk_modulate(x0)
    llr0 = compute_llr(y0, 0.1)
    if not np.array_equal(u0[info_t], sc_decode(llr0, frozen_t)[info_t]):
        err_sc += 1
assert err_sc == 0, f"SC 无损校验失败: {err_sc}/100"

# ========== 参数设置 ==========
QUICK = os.environ.get("POLAR_QUICK", "0") == "1"
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if QUICK else 100000
MIN_ERRORS = 5 if QUICK else 100
EB_N0_RANGE = np.arange(1.0, 4.0, 0.5) if QUICK else np.arange(0.0, 5.5, 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
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
        info_indices=info_idx,
        verbose=True,
        seed=42 + N,
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
