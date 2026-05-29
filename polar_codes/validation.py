"""模块正确性校验（各实验脚本导入）"""
import numpy as np

from encoder import polar_encode, build_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    """运行单元测试，失败时抛出 AssertionError"""
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    G = build_generator_matrix(4)
    x = polar_encode(u)
    x_mat = (u @ G) % 2
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"

    # SC 高信噪比无损
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errs = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u_full[info]):
            errs += 1
    assert errs == 0, f"SC 高 SNR 测试失败: {errs}/100 帧错误"

    # SCL L=1 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(20):
        u_full = np.zeros(N, dtype=int)
        u_full[info] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    if verbose:
        print("所有单元测试通过。")
