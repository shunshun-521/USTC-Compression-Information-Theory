"""单元测试：验证各模块正确性"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    """运行全部校验，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, x_expected), f"编码器错误: {x}"

    # SC 译码无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    frozen_bool = frozen.astype(bool)

    errors = 0
    for _ in range(100):
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = np.random.randint(0, 2, K)
        x_test = polar_encode(u_test)
        llr = np.where(x_test == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat, u_test):
            errors += 1
    assert errors == 0, f"SC 译码无损验证失败: {errors}/100 帧错误"

    errors = 0
    for _ in range(100):
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = np.random.randint(0, 2, K)
        x_test = polar_encode(u_test)
        sigma = eb_n0_to_sigma(12.0, 0.5)
        llr = compute_llr(
            bpsk_modulate(x_test) + np.random.normal(0, sigma, N), sigma
        )
        u_hat = sc_decode(llr, frozen_bool)
        if not np.array_equal(u_hat, u_test):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=12dB 下有 {errors}/100 帧错误"

    # 路径度量：L=1 的 SCL 应等价于 SC
    for _ in range(20):
        u_test = np.zeros(N, dtype=int)
        u_test[info_idx] = np.random.randint(0, 2, K)
        x_test = polar_encode(u_test)
        sigma = eb_n0_to_sigma(5.0, 0.5)
        llr = compute_llr(
            bpsk_modulate(x_test) + np.random.normal(0, sigma, N), sigma
        )
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    if verbose:
        print("所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_validation()
