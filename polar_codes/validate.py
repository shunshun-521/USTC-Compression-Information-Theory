"""极化码模块数值正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = np.where(x == 0, 100.0, -100.0)
    u_hat = sc_decode(llr, np.zeros(4, dtype=int))
    assert np.array_equal(u_hat, u), f"编码器/译码器往返失败: u={u}, x={x}, u_hat={u_hat}"
    print(f"编码器校验通过 (N=4, u={u}, x={x})")


def validate_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 仍有 {errors} 帧错误"
    print("SC 译码校验通过 (N=64, K=32, 100 帧 @ 10dB)")


def validate_scl_path_metric():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.1)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 与 SC 不等价"
    print("SCL 路径度量校验通过 (L=1 等价于 SC)")


def validate_crc():
    for _ in range(20):
        info = np.random.randint(0, 2, np.random.randint(8, 64))
        enc = crc_encode(info, 8)
        assert crc_check(enc, 8)
    print("CRC-8 校验通过")


def run_all_validations():
    validate_encoder()
    validate_sc_decoder()
    validate_scl_path_metric()
    validate_crc()
    print("全部单元测试通过。")


if __name__ == "__main__":
    run_all_validations()
