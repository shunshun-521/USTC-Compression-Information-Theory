"""单元测试与校验函数"""
import numpy as np
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有模块单元测试"""
    # 编码器环回校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = 100 * (1 - 2 * x)
    u_hat = sc_decode_recursive(llr, np.zeros(4, dtype=bool))
    assert np.array_equal(u_hat, u), f"编码器环回错误: {u_hat} != {u}"
    print("[PASS] 编码器环回校验")

    # SC 译码校验（Eb/N0=10dB, N=64, K=32）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码错误: {errors}/100 帧失败"
    print("[PASS] SC 译码校验 (100/100)")

    # 路径度量校验：L=1 SCL 应等价于 SC
    u_test = np.array([1, 0, 1, 1])
    llr_test = compute_llr(bpsk_modulate(polar_encode(u_test)), 0.1)
    frozen4 = np.zeros(4, dtype=bool)
    u_sc = sc_decode(llr_test, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), f"SCL L=1 不等价于 SC: {u_sc} vs {u_scl}"
    print("[PASS] SCL L=1 等价于 SC")

    print("所有单元测试通过！")


if __name__ == "__main__":
    run_unit_tests()
