"""模块正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def run_validations():
    """运行所有单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128: first 20 info indices = {info256[:20]}")

    # SC 译码校验（高 SNR）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在高 SNR 下出现 {sc_errors}/100 帧错误"

    # SCL L=1 应等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    scl_errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 与 SC 不一致: {scl_errors}/50 帧"

    # CRC 校验
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC-8 校验失败"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_validations()
