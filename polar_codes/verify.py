"""极化码模块单元测试"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    print("✓ 编码器测试通过")


def test_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"  N=8 info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"  N=256 info[:20]={info256[:20]}")
    print("✓ GA 构造测试通过")


def test_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(42)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 100 * bpsk_modulate(x)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损测试失败: {errors}/100 帧错误"
    print("✓ SC 无损译码测试通过")


def test_scl_equiv_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = 50 * bpsk_modulate(polar_encode(u))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"
    print("✓ SCL(L=1) 等价 SC 测试通过")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    coded[-1] ^= 1
    assert not crc_check(coded, 8)
    print("✓ CRC 测试通过")


def test_bp_noiseless():
    N, K = 32, 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    bp = BPDecoder(N, frozen, max_iter=50)

    u = np.zeros(N, dtype=int)
    u[info_idx] = 1
    llr = 100 * bpsk_modulate(polar_encode(u))
    u_hat, iters = bp.decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无损测试失败"
    print(f"✓ BP 无损译码测试通过 (iters={iters})")


if __name__ == "__main__":
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    test_encoder()
    test_construction()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_crc()
    test_bp_noiseless()
    print("\n所有测试通过！")
