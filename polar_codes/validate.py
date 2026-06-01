"""模块数值校验（各实验脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
    """编码器、SC、SCL 路径度量校验"""
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    x = polar_encode(u)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    # 标准参考向量 u=[0,0,1,1] -> x=[0,0,1,1]
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1])

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 高 SNR 译码失败"

    # SCL L=1 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    # CRC 自洽
    info = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    bad = coded.copy()
    bad[-1] ^= 1
    assert not crc_check(bad, 8)

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
