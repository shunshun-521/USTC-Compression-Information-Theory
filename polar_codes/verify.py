"""极化码模块单元测试（各实验脚本启动时调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行编码器、SC/SCL/BP 译码校验。"""
    if verbose:
        print("运行单元测试...")

    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {sc_errors} 个错误帧"

    llr0 = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), 1.0)
    u_sc = sc_decode(llr0, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr0)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"

    u_rec = sc_decode_recursive(llr0, frozen_bits)
    assert np.array_equal(u_sc, u_rec), "递归与非递归 SC 结果不一致"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print(f"GA N=8,K=4 info={info8}, frozen={frozen8}")

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print(f"GA N=256,K=128 first20 info={info256[:20]}")

    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 0]), 8)
    assert crc_check(bits, 8), "CRC 校验失败"

    if verbose:
        print("单元测试全部通过。\n")


if __name__ == "__main__":
    run_unit_tests()
