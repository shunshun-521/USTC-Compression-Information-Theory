"""模块数值校验（仿真脚本启动时调用）"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check, scl_equivalent_sc


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 1, 0, 1])

    msg = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    full = crc_encode(msg, 8)
    assert crc_check(full, 8)

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    rng = np.random.default_rng(123)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        if not np.array_equal(sc_decode(llr, frozen.astype(bool)), u):
            err += 1
    assert err == 0, f"SC 高信噪比测试失败: {err}/100 帧错误"

    llr0 = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), sigma)
    u_sc = sc_decode(llr0, frozen.astype(bool))
    u_scl = scl_equivalent_sc(llr0, frozen.astype(bool))
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    if verbose:
        print("[verify] 所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
