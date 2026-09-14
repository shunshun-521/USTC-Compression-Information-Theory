"""模块正确性校验"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_validation():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    print("编码器校验通过")

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info}, frozen={frozen}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, info前20: {info256[:20]}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[np.setdiff1d(np.arange(N), info_idx)] = True
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], bits):
            sc_errors += 1
        if not np.array_equal(u_hat, u_hat_r):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码校验失败: {sc_errors} 错误"
    print("SC 译码校验通过 (100 帧, Eb/N0=10dB)")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        bits = rng.integers(0, 2, K)
        u[info_idx] = bits
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc, _ = sc_decode(llr, frozen_bits), None
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 与 SC 不一致: {scl_errors}"
    print("SCL L=1 路径度量校验通过")

    bits8 = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    enc = crc_encode(bits8, 8)
    assert crc_check(enc, 8), "CRC 校验失败"
    print("CRC 校验通过")


if __name__ == "__main__":
    run_validation()
