"""极化码模块单元测试与数值校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器校验（与生成矩阵一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵不一致: {x} vs {x_mat}"

    u_check = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u_check), [0, 0, 1, 1]), "编码器蝶形结构错误"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("N=8, K=4, Eb/N0=2.5dB")
        print("info_indices:", info8)
        print("frozen_indices:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=256, K=128, first 20 info:", info256[:20])

    # SC 译码校验（高信噪比）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload), "SC 高信噪比译码失败"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(30):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    # CRC
    info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(info, 8)
    assert crc_check(enc, 8)

    if verbose:
        print("全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
