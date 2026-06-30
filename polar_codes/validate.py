"""极化码模块单元测试。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests():
    """运行全部单元测试，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码器与矩阵不一致: {x} vs {xm}"

    u4 = np.array([1, 0, 1, 1])
    x4 = polar_encode(u4)
    u4_dec = sc_decode(compute_llr(bpsk_modulate(x4), 1e-3), np.zeros(4, dtype=bool))
    assert np.array_equal(u4, u4_dec), f"N=4 往返失败: {u4_dec}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, 0.5)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(u_hat[info_idx], payload))
    assert ok == 100, f"SC 10dB 测试仅通过 {ok}/100 帧"

    u_test = np.zeros(N, dtype=np.int8)
    u_test[info_idx] = rng.integers(0, 2, size=K, dtype=np.int8)
    x_test = polar_encode(u_test)
    llr_test = compute_llr(bpsk_modulate(x_test), 1e-3)
    u_nr = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_test[info_idx], u_nr[info_idx]), "非递归 SC 噪声less失败"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    assert np.array_equal(u_scl, u_nr), "L=1 SCL 与 SC 不一致"

    info = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.append(info, [0, 0, 0, 0, 0, 0, 0, 1]), 8)

    print("全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
