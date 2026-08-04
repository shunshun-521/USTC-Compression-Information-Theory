"""
实验一：SC 译码基础仿真
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from decoder_sc import sc_decode
from simulation import run_simulation
from utils import save_results_csv, plot_bler_curves, save_frozen_set_info, find_capacity_limit
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

os.makedirs("results", exist_ok=True)


def run_validation():
    """单元测试：编码器与 SC 译码"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 验证 G@u（Arikan 核 F=[[1,1],[0,1]]）
    F = np.array([[1, 1], [0, 1]])
    G = np.kron(F, F)
    x_ref = np.zeros(4, dtype=int)
    for i in range(4):
        for j in range(4):
            x_ref[i] ^= G[i, j] * u[j]
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[np.setdiff1d(np.arange(N), info_idx)] = True
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x) + np.random.normal(0, sigma, N), sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_sent, u_rec):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors} 错误"
    print("单元测试通过。")


# ========== 参数设置 ==========
N_LIST = [256, 512]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 10000
MIN_ERRORS = 50
EB_N0_RANGE = np.arange(0.0, 5.5, 0.5)

if __name__ == "__main__":
    run_validation()

    save_frozen_set_info(N_LIST, None, DESIGN_EBN0, "results/frozen_sets.txt")

    all_results = {}
    for N in N_LIST:
        K = N // 2
        print(f"\n{'=' * 60}")
        print(f"SC 仿真: N={N}, K={K}, R={RATE}")
        print(f"{'=' * 60}")

        info_idx, frozen_idx, _ = ga_construction(N, K, DESIGN_EBN0)
        frozen_bits = np.zeros(N, dtype=bool)
        frozen_bits[frozen_idx] = True

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
