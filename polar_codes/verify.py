"""
单元测试与验证函数
"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_unit_tests(verbose=True):
    """运行所有模块单元测试。"""
    ok = True

    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"编码器错误: got {x}, expected {expected}")
    elif verbose:
        print("编码器校验通过")

    # SC 译码校验（极低噪声）
    rng = np.random.default_rng(42)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            sc_errors += 1

    if sc_errors > 0:
        ok = False
        if verbose:
            print(f"SC 译码校验失败: {sc_errors}/100 错误")
    elif verbose:
        print("SC 译码校验通过 (100 帧无误)")

    # SCL L=1 等价于 SC
    scl_mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_mismatch += 1

    if scl_mismatch > 0:
        ok = False
        if verbose:
            print(f"SCL L=1 与 SC 不一致: {scl_mismatch}/50")
    elif verbose:
        print("SCL L=1 路径度量校验通过")

    return ok


if __name__ == "__main__":
    success = run_unit_tests()
    raise SystemExit(0 if success else 1)
