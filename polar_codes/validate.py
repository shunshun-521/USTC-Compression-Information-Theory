"""各模块数值校验（实验脚本运行前调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, reorder_llr_for_decoder
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}"


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4 -> info:", info, "frozen:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info (first 20):", info256[:20])


def validate_sc_noiseless():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = reorder_llr_for_decoder(
            compute_llr(bpsk_modulate(polar_encode(u)), 1e-6), N
        )
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"


def validate_scl_equals_sc():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = reorder_llr_for_decoder(
            compute_llr(bpsk_modulate(polar_encode(u)), 1e-6), N
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 应等价于 SC"


def run_all():
    validate_encoder()
    validate_construction()
    validate_sc_noiseless()
    validate_scl_equals_sc()
    print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_all()
