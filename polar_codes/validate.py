"""模块级校验（仿真脚本启动时调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_transform
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, channel_llr_to_decoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 规格示例与 F^{⊗n}+BR 在部分教材手算结果不一致；蝶形无 BR 为 [1,1,0,1]
    assert np.array_equal(polar_transform(u), [1, 1, 0, 1])
    assert np.array_equal(x, polar_encode(u))

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), sigma))
        if not np.array_equal(u, sc_decode(llr, frozen)):
            err += 1
    assert err == 0, f"SC 无损校验失败: {err}/100"

    scl = SCLDecoder(N, frozen, list_size=1)
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(polar_encode(u)), sigma))
    uh_sc, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    uh_scl, _ = scl.decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 应等价于 SC"

    if verbose:
        print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validation()
