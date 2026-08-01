"""极化码仿真实验公共校验函数。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """运行所有模块的单元测试。"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造校验
    info, frozen, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"GA N=8 info={info}, frozen={frozen}")
    print(f"GA N=256 first 20 info={info256[:20]}")

    # SC 译码校验（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"

    # SCL L=1 等价于 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    # CRC 校验
    info_bits = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    encoded = crc_encode(info_bits, 8)
    assert crc_check(encoded, 8), "CRC 校验失败"

    print("所有单元测试通过。")
