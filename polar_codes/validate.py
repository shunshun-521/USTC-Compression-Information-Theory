"""极化码模块单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """数值正确性校验。"""
    print("运行单元测试...")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("  编码器校验通过")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * bpsk_modulate(x)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码无损验证失败 {sc_errors}/100 帧"
    print("  SC 译码校验通过")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 50.0 * bpsk_modulate(x)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_scl, u_sc):
            scl_errors += 1
    assert scl_errors == 0, f"L=1 SCL 与 SC 不一致 {scl_errors}/50 帧"
    print("  路径度量校验通过（L=1 SCL ≡ SC）")
    print("单元测试全部通过。\n")
