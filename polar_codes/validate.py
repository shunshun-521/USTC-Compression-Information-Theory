"""数值正确性校验"""
import numpy as np

from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_all_validations():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    design = 2.5
    info_idx, _, _ = ga_construction(N, K, design)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损验证失败，错误帧数={errors}"

    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    llr = np.where(x == 0, 50.0, -50.0)
    u_rec = sc_decode_recursive(llr, frozen_bits)
    u_sc = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_rec, u_sc), "递归与非递归 SC 不一致"

    u_hat_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_hat_scl, u_sc), "L=1 SCL 与 SC 不一致"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_all_validations()
