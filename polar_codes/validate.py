"""极化码模块数值正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import (
    bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, bit_reverse_llr,
)
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    if verbose:
        print(f"[OK] 编码器: u={u} -> x={x}")

    # SC 译码无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = bit_reverse_llr(compute_llr(y, sigma))
        if not np.array_equal(u, sc_decode(llr, frozen)):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码 @10dB 错误帧数: {sc_errors}/100"
    if verbose:
        print("[OK] SC 译码 @10dB: 100/100 正确")

    # SCL L=1 等价 SC
    scl_mismatch = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(5.0, K / N)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = bit_reverse_llr(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            scl_mismatch += 1
    assert scl_mismatch == 0, f"SCL L=1 与 SC 不一致: {scl_mismatch}/50"
    if verbose:
        print("[OK] SCL L=1 等价 SC")

    # CRC 校验
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(info, 8)
    assert crc_check(encoded, 8), "CRC 编码/校验失败"
    if verbose:
        print("[OK] CRC-8 编码与校验")

    if verbose:
        print("全部单元测试通过。")


if __name__ == '__main__':
    run_unit_tests()
