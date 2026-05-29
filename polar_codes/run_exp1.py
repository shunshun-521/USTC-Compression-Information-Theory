"""
实验一：SC 译码基础仿真
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit

os.makedirs("results", exist_ok=True)

FAST = os.environ.get("POLAR_FAST", "0") == "1"
MAX_FRAMES = 2000 if FAST else 100000
MIN_ERRORS = 30 if FAST else 100

# ========== 单元测试 ==========
u = np.array([1, 0, 1, 1])
x = polar_encode(u)
x_exp = (u @ build_generator_matrix(4)) % 2
assert np.array_equal(x, x_exp), f"编码器错误: {x} vs {x_exp}"

N, K = 64, 32
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
rng = np.random.default_rng(0)
err = 0
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

sigma = eb_n0_to_sigma(10.0, K / N)
for _ in range(100):
    u_t = np.zeros(N, dtype=int)
    u_t[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u_t)), sigma, rng), sigma)
    if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u_t[info_idx]):
        err += 1
assert err == 0, f"SC 高信噪比校验失败: {err}/100"

from decoder_scl import SCLDecoder

scl = SCLDecoder(N, frozen_bits, list_size=1)
for _ in range(20):
    u_t = np.zeros(N, dtype=int)
    u_t[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_t)), 1e-12)
    u1, _ = scl.decode(llr)
    u2 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1, u2)
print("单元测试通过。")

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

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
        N=N, K=K, eb_n0_db_list=EB_N0_RANGE, decoder=decoder,
        decoder_type="sc", max_frames=MAX_FRAMES, min_errors=MIN_ERRORS,
        info_indices=info_idx, frozen_bits=frozen_bits, design_eb_n0_db=DESIGN_EBN0,
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
