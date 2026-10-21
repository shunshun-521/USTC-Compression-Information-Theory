"""
极化码 SC（串行抵消）译码器
非递归实现（置换 SC，Vangala et al. 2014）
"""
import math
import os
import sys
from contextlib import contextmanager

import numpy as np


@contextmanager
def _sc_ref_imports():
    sc_ref_path = os.path.join(os.path.dirname(__file__), "sc_ref")
    inserted = False
    if sc_ref_path not in sys.path:
        sys.path.insert(0, sc_ref_path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(sc_ref_path)


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    """min-sum f（SCL/BP 模块使用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    if np.isscalar(u_hat) or (isinstance(u_hat, np.ndarray) and u_hat.ndim == 0):
        b = int(u_hat)
        return (La + Lb) if b == 0 else (La - Lb)
    u_hat = np.asarray(u_hat, dtype=int)
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    with _sc_ref_imports():
        from decoder_utils import active_bit_level, active_llr_level

    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    with _sc_ref_imports():
        from SCD import SCD

        class _PC:
            pass

        pc = _PC()
        pc.N = N
        pc.n = n
        pc.frozen = frozen_set
        pc.likelihoods = llr_ch
        return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
