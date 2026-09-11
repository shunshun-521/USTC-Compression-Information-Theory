"""
极化码模块数值正确性校验
"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def _arikan_gen(n):
    F = np.array([[1, 1], [0, 1]], dtype=int)
    Fn = F
    for _ in range(n - 1):
        Fn = np.kron(F, Fn)
    return Fn


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    n = int(np.log2(len(u)))
    expected = (u @ _arikan_gen(n)) % 2
    assert np.array_equal(x, expected), f"编码器错误: got {x}, expected {expected}"
    print("✓ 编码器校验通过")


def validate_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    expected_info = np.array([0, 3, 5, 6])
    expected_frozen = np.array([1, 2, 4, 7])
    assert np.array_equal(info, expected_info), f"GA N=8 info错误: {info}"
    assert np.array_equal(frozen, expected_frozen), f"GA N=8 frozen错误: {frozen}"
    print("✓ GA 构造校验通过 (N=8, K=4)")


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 100.0
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("✓ SC 无损译码校验通过 (N=64, K=32, Eb/N0=10dB)")


def validate_sc_recursive():
    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    for _ in range(20):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_nr = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_nr), "递归与非递归 SC 不一致"
    print("✓ SC 递归/非递归一致性校验通过")


def validate_scl_equals_sc():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(2)
    sigma = eb_n0_to_sigma(6.0, K / N)
    for _ in range(50):
        info = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("✓ SCL(L=1) 等价 SC 校验通过")


def run_all():
    print("=" * 50)
    print("极化码模块校验")
    print("=" * 50)
    validate_encoder()
    validate_ga_construction()
    validate_sc_noiseless()
    validate_sc_recursive()
    validate_scl_equals_sc()
    print("=" * 50)
    print("所有校验通过!")
    print("=" * 50)


if __name__ == "__main__":
    run_all()
