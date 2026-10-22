"""单元测试：编码器与各译码器正确性校验"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_validations(verbose=True):
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(12.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors} 帧错误"

    # 非递归 SC 在随机 LLR 下应产生合法比特
    llr_test = rng.normal(0, 2, size=N)
    un = sc_decode(llr_test, frozen_bits)
    assert un.shape == (N,) and set(un).issubset({0, 1})

    # L=1 SCL 等价 SC
    llr_test = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), 0.5, rng=rng), 0.5)
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_all_validations()
