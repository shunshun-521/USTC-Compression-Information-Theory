"""模块数值校验（实验脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    # 编码器：与生成矩阵 G 一致
    G = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.mod(u @ G, 2)), f"编码器错误: {x}"

    u2 = np.array([0, 1, 0, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1]), f"编码器参考码字错误: {polar_encode(u2)}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("N=8 GA info:", info8, "frozen:", frozen8)

    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=256 info first 20:", info256[:20])

    # SC 高信噪比
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)) * 20.0, sigma)
        if not np.array_equal(sc_decode(llr, frozen)[info], payload):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err} 帧错误"

    # SCL L=1 等价 SC
    rng = np.random.default_rng(1)
    for _ in range(20):
        payload = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)) * 20.0, sigma)
        uh_sc = sc_decode(llr, frozen)
        uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(uh_sc, uh_scl)

    if verbose:
        print("validate.py: 全部检查通过。")
    return True


if __name__ == "__main__":
    run_validation()
