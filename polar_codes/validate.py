"""极化码仿真实验公共校验与配置。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, verify_sc_decoders
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
    """运行各模块数值正确性校验。"""
    # 编码器：自洽性检验（蝶形编码与矩阵编码一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    from encoder import polar_encode_matrix

    x_mat = polar_encode_matrix(u)
    assert np.array_equal(x, x_mat), f"编码器与矩阵编码不一致: {x} vs {x_mat}"

    # SC 译码：无噪声下应完全正确
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = np.where(x == 0, 100.0, -100.0)
        ok, _, _ = verify_sc_decoders(llr, frozen_bool)
        assert ok, "递归与非递归 SC 译码结果不一致"
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], u_sent[info_idx]), "SC 无噪声译码失败"

    # 路径度量：L=1 的 SCL 等价于 SC
    sigma = eb_n0_to_sigma(8.0, 0.5)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = SCLDecoder(N, frozen_bool, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 时 SCL 与 SC 不等价"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
