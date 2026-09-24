"""单元测试：验证各模块正确性"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print(f"  编码器: u={u} -> x={x} (与 G_N 一致)")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print(f"  GA 构造 N=8: info={info}, frozen={frozen}")


def test_sc_noiseless():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
    print("  SC 无损译码: 100/100 帧正确")


def test_sc_nonrecursive():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        u_nonrec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec, u_nonrec), "递归与非递归 SC 不一致"
    print("  递归与非递归 SC 一致")


def test_scl_equiv_sc():
    N = 32
    K = 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(2)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
    print("  L=1 SCL 等价于 SC")


def test_crc():
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8), "CRC 应检测到错误"
    print("  CRC 编解码正确")


def test_bp_noiseless():
    N = 32
    K = 16
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    bp = BPDecoder(N, frozen_bits, max_iter=50)

    rng = np.random.default_rng(3)
    errors = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat, iters = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"BP 无损译码失败: {errors}/20"
    print(f"  BP 无损译码: 20/20 帧正确")


def main():
    tests = [
        ("编码器", test_encoder),
        ("GA 构造", test_ga_construction),
        ("SC 无损译码", test_sc_noiseless),
        ("SC 递归/非递归一致性", test_sc_nonrecursive),
        ("SCL L=1 等价 SC", test_scl_equiv_sc),
        ("CRC", test_crc),
        ("BP 无损译码", test_bp_noiseless),
    ]
    print("=" * 50)
    print("极化码模块验证")
    print("=" * 50)
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    print("=" * 50)
    print(f"结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
