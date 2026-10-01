"""
实验一：SC 译码基础仿真
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit
from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from validate import run_all as run_validate

os.makedirs("results", exist_ok=True)

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
assert np.array_equal(x, polar_encode_matrix(u)), f"编码器错误: {x}"

N_t, K_t = 64, 32
info_t, _, _ = ga_construction(N_t, K_t, 2.5)
frozen_t = np.ones(N_t, dtype=int)
frozen_t[info_t] = 0
rng = np.random.default_rng(0)
sigma_t = eb_n0_to_sigma(10.0, 0.5)
sc_pass = 0
for _ in range(100):
    payload = rng.integers(0, 2, K_t)
    uu = np.zeros(N_t, dtype=int)
    uu[info_t] = payload
    y = awgn_channel(bpsk_modulate(polar_encode(uu)), sigma_t, rng)
    uh = sc_decode(compute_llr(y, sigma_t), frozen_t)
    sc_pass += int(np.array_equal(uh[info_t], payload))
assert sc_pass >= 90, f"SC 高信噪比校验通过率过低: {sc_pass}/100"

run_validate()

FAST = os.environ.get("POLAR_FAST_SIM", "0") == "1"

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 500 if FAST else 100000
MIN_ERRORS = 10 if FAST else 100
EB_N0_RANGE = np.arange(1.0, 3.5, 0.5) if FAST else np.arange(0.0, 5.5, 0.25)

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
        frozen_bits=frozen_bits,
        info_indices=info_idx,
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
