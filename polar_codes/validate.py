"""模块数值校验（实验脚本启动前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    if verbose:
        print("GA N=8,K=4 info:", info8, "frozen:", frozen8)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    err = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            err += 1
    assert err == 0, f"SC 无损译码失败: {err}/100"

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u = np.zeros(N, dtype=np.int8)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    uh_sc = sc_decode(llr, frozen_bits)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 时 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 所有校验通过")


if __name__ == "__main__":
    run_validation()
