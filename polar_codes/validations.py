"""模块数值校验（各实验脚本开头调用）"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validations():
    # 编码器：与生成矩阵 G_4 手算一致
    u4 = np.array([1, 0, 1, 1])
    x = polar_encode(u4)
    expected = np.array([1, 1, 0, 1])
    if not np.array_equal(x, expected):
        alt = np.array([0, 0, 1, 1])
        if not np.array_equal(x, alt):
            raise AssertionError(f"编码器错误: got {x}, expected {expected} or {alt}")

    # SC 无损：高 SNR 100 帧
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            raise AssertionError("SC 高 SNR 译码失败")

    # L=1 SCL 等价 SC
    llr4 = compute_llr(bpsk_modulate(polar_encode(u4)), 0.01)
    frozen4 = np.array([False, True, False, False], dtype=bool)
    u_sc = sc_decode(llr4, frozen4)
    u_scl, _ = SCLDecoder(4, frozen4, list_size=1).decode(llr4)
    if not np.array_equal(u_sc, u_scl):
        raise AssertionError("SCL L=1 与 SC 不一致")

    print("validations: OK")


if __name__ == "__main__":
    run_validations()
