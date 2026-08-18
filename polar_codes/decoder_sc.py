"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from _scd_impl2 import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    logdomain_sum,
)
from _scd_local import SCDLocal
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def f_operation_exact(La, Lb):
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation_exact(La, Lb, u_hat):
    if np.isscalar(u_hat):
        return La + Lb if u_hat == 0 else La - Lb
    return (1.0 - 2.0 * u_hat) * La + Lb


def bit_reversed_index(x, n):
    return bit_reversed(x, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，输入为重排后的 LLR）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        decode_node(f_operation_exact(llr_node[:half], llr_node[half:]), bit_offset)
        decode_node(
            g_operation_exact(
                llr_node[:half], llr_node[half:], u_hat[bit_offset:bit_offset + half]
            ),
            bit_offset + half,
        )

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - active_llr_level(phi, n), n)))
        bit_layer_vec.append(
            list(range(n - active_bit_level(phi, n), n)) if phi % 2 == 1 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _frozen_indices_from_mask(frozen_bits):
    return np.where(np.asarray(frozen_bits).astype(bool))[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]
    frozen_indices = _frozen_indices_from_mask(frozen_bits)

    pc = type(
        "PolarCodeView",
        (),
        {
            "N": N,
            "n": n,
            "K": N - len(frozen_indices),
            "frozen": frozen_indices,
            "likelihoods": llr,
        },
    )()
    return SCDLocal(pc).decode()
