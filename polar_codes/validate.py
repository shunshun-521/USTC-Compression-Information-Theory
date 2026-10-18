"""模块数值校验（各实验脚本启动时调用）。"""
import numpy as np

from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    info8, fr8, _ = ga_construction(8, 4, 2.5)
    info256, _, _ = ga_construction(256, 128, 2.5)
    if verbose:
        print("N=8 info:", info8, "frozen:", fr8)
        print("N=256 info (first 20):", info256[:20])

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.zeros(N, dtype=bool)
    frozen[np.setdiff1d(np.arange(N), info)] = True
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1 - 2 * x) * 100.0
        uh = sc_decode(llr, frozen)
        if np.all(uh[info] == u[info]):
            ok += 1
    assert ok == 100, f"SC 无损译码测试失败: {ok}/100"

    frozen_i = frozen.astype(int)
    llr = (1 - 2 * polar_encode(np.zeros(N, int))) * 50.0
    u_sc, _ = SCLDecoder(N, frozen_i, list_size=1, crc_length=0).decode(llr)
    u_rec = sc_decode(llr, frozen)
    assert np.array_equal(u_sc, u_rec), "L=1 SCL 应与 SC 一致"

    if verbose:
        print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_validation()
