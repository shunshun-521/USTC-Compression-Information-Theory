"""极化码编译码仿真公共单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有模块单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        payload = rng.integers(0, 2, size=K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在高信噪比下失败 {sc_errors}/100 帧"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_errors = 0
    for _ in range(50):
        payload = rng.integers(0, 2, size=K)
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = payload
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_errors += 1
    assert scl_errors == 0, f"SCL(L=1) 与 SC 不一致 {scl_errors}/50 帧"
    print("所有单元测试通过。")
