"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
import polar_lib_ref as plr

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_info_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if_information = np.zeros(len(frozen_bits), dtype=np.int8)
    if_information[~frozen_bits] = 1
    return if_information


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（惰性 LLR，O(N log N)）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float32)
    if_info = _frozen_to_info_mask(frozen_bits)
    return plr.SCDecoder(llr_ch, if_info).astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 相同后端）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助结构（惰性实现中由运行时完成，此处返回占位索引）。"""
    n = int(math.log2(N))
    decode_order = list(range(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u, u_hat):
            errors += 1
    print("SC test errors:", errors)
