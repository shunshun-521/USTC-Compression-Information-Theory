"""极化码仿真公共单元测试"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    """运行所有模块单元测试，失败时抛出 AssertionError"""
    # 编码器：蝶形结构自洽性（u -> x -> 噪声less 解码）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4 and set(x).issubset({0, 1}), f"编码器输出非法: {x}"

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    frozen_bool = frozen.astype(bool)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat, u_sent), "SC 译码在高信噪比下失败"

    # SCL L=1 等价于 SC
    scl = SCLDecoder(N, frozen_bool, list_size=1)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不等价"

    # CRC 测试
    info = rng.integers(0, 2, 16, dtype=np.int8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 校验失败"

    if verbose:
        print("所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
