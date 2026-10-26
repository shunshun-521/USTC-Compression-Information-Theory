"""单元测试：编码器、SC/SCL、路径度量"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_validation(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    if not np.array_equal(x, polar_encode_matrix(u)):
        ok = False
        if verbose:
            print(f"编码器与矩阵不一致: {x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 1e3 * (1 - 2 * x)
        if not np.array_equal(sc_decode(llr, frozen), u):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"SC 高信噪比测试失败: {sc_err}/100 帧错误")

    scl = SCLDecoder(N, frozen, list_size=1)
    scl_err = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 1e3 * (1 - 2 * x)
        uh, _ = scl.decode(llr)
        uh_sc = sc_decode_recursive(llr, frozen)
        if not np.array_equal(uh, uh_sc):
            scl_err += 1
    if scl_err > 0:
        ok = False
        if verbose:
            print(f"SCL L=1 与 SC 不一致: {scl_err}/50")

    if verbose:
        print("单元测试通过" if ok else "单元测试存在失败项")
    return ok


if __name__ == "__main__":
    run_validation()
