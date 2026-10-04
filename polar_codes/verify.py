"""模块数值校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_decode_paths
from encoder import polar_encode, polar_encode_matrix


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败时抛出 AssertionError"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
    assert np.array_equal(x, polar_encode_matrix(u)), "编码器与矩阵法不一致"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        uh = sc_decode(llr, frozen, use_min_sum=False)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            err += 1
    assert err == 0, f"SC 高 SNR 测试失败: {err}/100 帧错误"

    paths_sc = scl_decode_paths(llr, frozen, list_size=1, use_min_sum=False)
    paths_scl = scl_decode_paths(llr, frozen, list_size=1, use_min_sum=False)
    assert np.array_equal(paths_sc[0][1], paths_scl[0][1]), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("全部单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
