"""模块数值正确性校验（仿真脚本可调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    G = polar_generator_matrix(4)
    x = polar_encode(u)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    ok = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u_full)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, 0.5))
        u_hat = sc_decode(llr, frozen_bits)
        ok += int(np.array_equal(u_hat, u_full))
    assert ok == 100, f"SC 无损测试失败: {ok}/100"

    # L=1 SCL 等价 SC
    llr = compute_llr(bpsk_modulate(polar_encode(u_full)), eb_n0_to_sigma(10.0, 0.5))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("verify.py: 全部校验通过")


if __name__ == "__main__":
    run_unit_tests()
