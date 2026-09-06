"""单元测试与模块验证"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_encode_matrix(4)
    x_mat = u @ G % 2
    assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
    print("编码器校验通过:", u, "->", x)


def test_sc_noiseless(N=64, K=32, num_frames=100):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for dec_name, dec_fn in [
        ("recursive", sc_decode_recursive),
        ("non-recursive", sc_decode),
    ]:
        for _ in range(num_frames):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            llr = compute_llr(bpsk_modulate(x), sigma)
            u_hat = dec_fn(llr, frozen_bits.astype(bool))
            assert np.array_equal(u_hat[info_idx], u[info_idx]), f"{dec_name} SC 译码错误"
        print(f"SC ({dec_name}) 无损校验通过: N={N}, {num_frames} 帧")


def test_scl_equiv_sc(N=64, K=32, num_frames=20):
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 等价"
    print(f"SCL(L=1) 路径度量校验通过: {num_frames} 帧")


def test_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits, 8)
    assert crc_check(coded, 8)
    print("CRC 校验通过")


if __name__ == "__main__":
    test_encoder()
    test_crc()
    test_sc_noiseless()
    test_scl_equiv_sc()
