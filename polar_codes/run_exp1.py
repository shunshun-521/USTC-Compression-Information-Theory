"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
- 仿真并绘制 BLER-Eb/N0 曲线
- 添加 BPSK 信道容量限
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from channel import eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr
from simulation import run_simulation, quick_env_defaults
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_unit_tests():
    """数值正确性校验。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)
    rng = np.random.default_rng(0)
    sigma = 1e-3  # 极低噪声，验证译码器而非构造 BLER
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info
        x = polar_encode(u_sent)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], info), "SC 译码失败"
        u_hat_r = sc_decode_recursive(llr, frozen_bool)
        assert np.array_equal(u_hat, u_hat_r), "递归/非递归 SC 不一致"

    print("单元测试通过。")


os.makedirs("results", exist_ok=True)
run_unit_tests()

MAX_FRAMES, MIN_ERRORS = quick_env_defaults()
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt", rate=RATE)

all_results = {}

for N in N_LIST:
    K = N // 2
    print(f"\n{'=' * 60}")
    print(f"SC 仿真: N={N}, K={K}, R={RATE}")
    print(f"{'=' * 60}")

    info_idx, _, _ = ga_construction(N, K, DESIGN_EBN0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    def decoder(llr_ch, _fb=frozen_bits.astype(bool)):
        return sc_decode(llr_ch, _fb), None

    results = run_simulation(
        N=N, K=K,
        eb_n0_db_list=EB_N0_RANGE,
        decoder=decoder,
        decoder_type="sc",
        max_frames=MAX_FRAMES,
        min_errors=MIN_ERRORS,
        info_indices=info_idx,
        frozen_bits=frozen_bits.astype(bool),
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
