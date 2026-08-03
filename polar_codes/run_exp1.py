"""
实验一：SC 译码基础仿真
- 码长 N = 256, 512, 1024
- 码率 R = 1/2
- GA 构造，设计 Eb/N0 = 2.5 dB
- 仿真并绘制 BLER-Eb/N0 曲线
- 添加 BPSK 信道容量限
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit


def run_validation_tests():
    """数值正确性校验"""
    print("=" * 40)
    print("运行单元测试...")
    print("=" * 40)

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = int(np.log2(len(u)))
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((len(u), len(u)), dtype=int)
    for i in range(len(u)):
        j = int(format(i, f'0{n}b')[::-1], 2)
        B[i, j] = 1
    assert np.array_equal(x, (u @ ((B @ G) % 2)) % 2), f"编码器错误: {x}"
    print("  [PASS] 编码器校验")

    # SC 译码校验（无损验证）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, size=K)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = info_bits
        x = polar_encode(u_sent)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 个错误"
    print("  [PASS] SC 译码无损校验 (N=64, 100帧)")

    # 递归与非递归 SC 一致性
    u_test = rng.integers(0, 2, size=N)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = u_test[:K] if len(u_test) >= K else rng.integers(0, 2, size=K)
    u_sent[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u_sent)
    s = bpsk_modulate(x)
    y = awgn_channel(s, 0.1, rng)
    llr = compute_llr(y, 0.1)
    u_rec = sc_decode(llr, frozen_bits)
    u_rec2 = sc_decode_recursive(llr, frozen_bits.astype(bool))
    assert np.array_equal(u_rec, u_rec2), "递归与非递归 SC 不一致"
    print("  [PASS] 递归/非递归 SC 一致性")

    print("所有单元测试通过!\n")


os.makedirs('results', exist_ok=True)

# ========== 参数设置 ==========
N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

if __name__ == "__main__":
    run_validation_tests()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, 'results/frozen_sets.txt')

    all_results = {}

    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'=' * 60}")

        info_idx, frozen_idx, llr_means = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits), None

        results = run_simulation(
            N=N, K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type='sc',
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            verbose=True,
            info_indices=info_idx,
            frozen_bits=frozen_bits,
        )

        label = f'SC, N={N}, K={K}'
        all_results[label] = results
        save_results_csv(results, f'results/exp1_sc_N{N}_R0.5.csv')

    shannon_db = find_capacity_limit(RATE)
    print(f"\nBPSK 信道容量限（R={RATE}）: Eb/N0 = {shannon_db:.3f} dB")

    plot_bler_curves(
        all_results,
        title=f'SC Decoder BLER vs Eb/N0 (R={RATE})',
        save_path='results/fig1_sc_bler.png',
        shannon_limit_db=shannon_db,
    )
    print("\n实验一完成。结果保存至 results/ 目录。")
