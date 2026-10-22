"""各模块数值校验（实验脚本启动前调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError。"""
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # SC 无损（10 dB，N=64, K=32）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码失败"

    # 非递归与递归一致
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    assert np.array_equal(
        sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits)
    )

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen_bits.astype(int), list_size=1, crc_length=0)
    u_hat_scl, _ = scl.decode(llr)
    assert np.array_equal(u_hat_scl, sc_decode(llr, frozen_bits))

    if verbose:
        print("validate.py: 全部校验通过。")


if __name__ == "__main__":
    run_validation()
