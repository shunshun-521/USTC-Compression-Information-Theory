"""极化码模块数值校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G 矩阵不一致: {x} vs {x_ref}"
    print("编码器校验通过:", x.tolist())


def test_sc_noiseless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info], u[info]), "SC 无损译码失败"
    print("SC 高 SNR 校验通过 (100 帧)")


def test_scl_l1_equals_sc():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    scl = SCLDecoder(N, frozen, list_size=1)
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        assert np.array_equal(sc_decode(llr, frozen), scl.decode(llr)[0])
    print("SCL L=1 与 SC 等价校验通过")


def test_ga_reference():
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print("GA N=8 info:", info8, "frozen:", frozen8)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("GA N=256 info (first 20):", info256[:20])


if __name__ == "__main__":
    test_encoder()
    test_sc_noiseless()
    test_scl_l1_equals_sc()
    test_ga_reference()
    print("\n全部校验通过。")
