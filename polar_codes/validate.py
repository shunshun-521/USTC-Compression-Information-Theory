"""模块数值校验（实验脚本运行前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests(verbose=True):
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # SC 无损
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_err += 1
    assert sc_err == 0, f"SC 译码在高信噪比下失败 {sc_err}/100 帧"

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen_bits, list_size=1, info_indices=info_idx)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        uh, _ = scl.decode(llr)
        assert np.array_equal(uh, sc_decode(llr, frozen_bits))

    # CRC
    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
    assert crc_check(bits, 8)

    if verbose:
        print("validate.py: 全部单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
