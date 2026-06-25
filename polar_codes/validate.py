"""极化码模块单元测试"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器：矩阵乘法验证
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    # GA 构造
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4
    assert len(info256) == 128

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "SC 译码失败"

    # 递归与非递归一致
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-6)
    assert np.array_equal(
        sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits)
    ), "递归/非递归 SC 不一致"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_hat_scl, _ = scl.decode(llr)
    assert np.array_equal(u_hat_scl, sc_decode(llr, frozen_bits)), "SCL L=1 与 SC 不一致"

    # CRC 自检
    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    if verbose:
        print("全部单元测试通过。")
        print(f"N=8,K=4 info={info8}, frozen={frozen8}")
        print(f"N=256,K=128 info[:20]={info256[:20]}")


if __name__ == "__main__":
    run_unit_tests()
