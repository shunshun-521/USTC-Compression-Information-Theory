"""模块数值校验（各实验脚本开头调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def polar_generator_matrix(N):
    """由 polar_encode 逐比特基向量生成 G（行向量 u @ G）。"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        u = np.zeros(N, dtype=int)
        u[i] = 1
        G[i] = polar_encode(u)
    return G


def run_unit_tests(verbose=True):
    N = 4
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(N)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器与 G 矩阵不一致: {x} vs {x_ref}"
    assert np.array_equal(x, np.array([1, 1, 0, 1])), f"N=4 编码参考: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    errs = 0
    for _ in range(100):
        ut = np.zeros(N, dtype=int)
        ut[info] = rng.integers(0, 2, K)
        codeword = polar_encode(ut)
        y = awgn_channel(bpsk_modulate(codeword), sigma, rng)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if not np.array_equal(uh[info], ut[info]):
            errs += 1
    assert errs == 0, f"SC 高信噪比校验失败: {errs} 帧错误"

    info8, _, _ = ga_construction(8, 4, 2.5)
    frozen8 = np.ones(8, dtype=int)
    frozen8[info8] = 0
    llr0 = np.ones(8) * 5.0
    u_sc = sc_decode(llr0, frozen8)
    scl = SCLDecoder(8, frozen8, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr0)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
