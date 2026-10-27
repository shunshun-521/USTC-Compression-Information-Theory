"""数值正确性校验（各实验脚本运行前调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"


def validate_sc_lossless():
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    eb_db = 10.0
    sigma = eb_n0_to_sigma(eb_db, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无损译码失败"


def validate_scl_equals_sc():
    N, K = 128, 64
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(7)
    sigma = eb_n0_to_sigma(8.0, 0.5)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = scl.decode(llr)
        assert np.array_equal(uh_sc, uh_scl), "L=1 的 SCL 应与 SC 等价"


def run_all():
    validate_encoder()
    validate_sc_lossless()
    validate_scl_equals_sc()
    print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_all()
