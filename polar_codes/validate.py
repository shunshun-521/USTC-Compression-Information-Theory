"""单元测试与数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"
    print("  [PASS] 编码器校验")


def test_ga_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    assert len(info) == 4 and len(frozen) == 4
    assert len(set(info) | set(frozen)) == 8
    print("  [PASS] GA 构造校验")


def test_sc_noiseless(N=64, K=32, num_frames=100, eb_n0_db=20.0):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)

    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        # 极低噪声验证（等效无噪声）
        y = bpsk_modulate(x) + rng.normal(0, 1e-6, N)
        llr = compute_llr(y, 1e-6)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码失败"

    print(f"  [PASS] SC 译码校验 (N={N}, {num_frames} 帧)")


def test_scl_equiv_sc(N=64, K=32):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(4.0, rate)

    for _ in range(20):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("  [PASS] SCL L=1 等价 SC")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    print("  [PASS] CRC 校验")


def test_bp_roundtrip(N=16, K=8):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat, _ = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
    assert np.array_equal(u_hat[info_idx], u[info_idx])
    print("  [PASS] BP 无噪声校验")


def run_unit_tests():
    print("运行单元测试...")
    test_encoder()
    test_ga_construction()
    test_crc()
    test_sc_noiseless()
    test_scl_equiv_sc()
    test_bp_roundtrip()
    print("所有单元测试通过。\n")


if __name__ == "__main__":
    run_unit_tests()
