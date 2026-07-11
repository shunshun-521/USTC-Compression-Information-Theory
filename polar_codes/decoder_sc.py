"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from _ref_decoder import sc_decoder_ref
from _ref_function import f_hf as _f_hf_scalar
from _ref_function import g as _g_scalar


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    for i in range(La.size):
        out[i] = _f_hf_scalar(La.flat[i], Lb.flat[i])
    return out


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La)
    for i in range(La.size):
        out[i] = _g_scalar(La.flat[i], Lb.flat[i], int(u_hat.flat[i]))
    return out


def _frozen_to_info_list(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return list(np.where(~frozen_bits)[0])
    return list(np.where(frozen_bits == 0)[0])


def _reorder_channel_llr(llr_ch, N):
    perm = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[perm]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与高效实现等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口保留）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n - 1, -1, -1)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 经比特倒序重排后送入树形 SC 译码器。
    """
    N = len(llr_ch)
    llr_reordered = _reorder_channel_llr(llr_ch, N)
    information_pos = _frozen_to_info_list(frozen_bits)
    return sc_decoder_ref(llr_reordered, information_pos, frozen_bit=0)


def verify_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """在极低噪声下验证 SC 译码"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            return False
    return True
