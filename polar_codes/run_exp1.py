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

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from encoder import polar_encode, polar_encode_matrix
from simulation import run_simulation
from utils import (
    find_capacity_limit,
    plot_bler_curves,
    save_frozen_set_info,
    save_results_csv,
)


def run_validation_tests():
    """数值正确性校验"""
    print("=" * 50)
    print("运行单元测试...")
    print("=" * 50)

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = polar_encode_matrix(u)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"
    print(f"  编码器校验通过: u={u} -> x={x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, K)
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = info
        llr = compute_llr(bpsk_modulate(polar_encode(u_test)), sigma)
        u_hat = sc_decode(llr, frozen.astype(bool))
        if not np.array_equal(u_hat[info_idx], info):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码校验失败: {sc_errors}/100 帧错误"
    print(f"  SC 译码校验通过: 100 帧 Eb/N0=10dB 全部正确")

    from decoder_scl import SCLDecoder

    scl = SCLDecoder(N, frozen.astype(bool), list_size=1)
    scl_errors = 0
    for _ in range(50):
        info = rng.integers(0, 2, K)
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = info
        llr = compute_llr(bpsk_modulate(polar_encode(u_test)), sigma)
        u_sc = sc_decode(llr, frozen.astype(bool))
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 与 SC 不等价: {scl_errors}/50"
    print("  路径度量校验通过: SCL(L=1) 等价于 SC")
    print("所有单元测试通过!\n")


os.makedirs("results", exist_ok=True)

N_LIST = [256, 512, 1024]
RATE = 0.5
DESIGN_EBN0 = 2.5
MAX_FRAMES = 100000
MIN_ERRORS = 100
EB_N0_RANGE = np.arange(0.0, 5.5, 0.25)

if __name__ == "__main__":
    run_validation_tests()

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

        def decoder(llr_ch):
            return sc_decode(llr_ch, frozen_bits.astype(bool)), None

        results = run_simulation(
            N=N,
            K=K,
            eb_n0_db_list=EB_N0_RANGE,
            decoder=decoder,
            decoder_type="sc",
            max_frames=MAX_FRAMES,
            min_errors=MIN_ERRORS,
            info_indices=info_idx,
            design_eb_n0_db=DESIGN_EBN0,
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
