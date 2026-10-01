"""模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def run_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 info first 20:", info256[:20])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    sc_ok = 0
    for _ in range(200):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat_r, u_hat), "SC 递归与非递归不一致"
        if np.array_equal(u_hat[info_idx], payload):
            sc_ok += 1
    assert sc_ok >= 190, f"SC 高信噪比通过率过低: {sc_ok}/200"

    scl8 = SCLDecoder(N, frozen_bits, list_size=8)
    ok = 0
    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u8, _ = scl8.decode(llr)
        if np.array_equal(u8[info_idx], payload):
            ok += 1
    assert ok >= 45, f"SCL L=8 高信噪比通过率 {ok}/50"

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    print("validate.py: 全部检查通过")


if __name__ == "__main__":
    run_all()
