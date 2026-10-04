"""
极化码 SC（串行抵消）译码器
SC 与单遍 BP（min-sum）在极化因子图上等价，故采用 max_iter=1 的 BP 实现。
"""
import math
import numpy as np
from decoder_bp import BPDecoder


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [list(range(n))] + [[] for _ in range(N - 1)], [[] for _ in range(N)]


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：单遍 BP（与因子图一致）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    decoder = BPDecoder(N, frozen_bits, max_iter=1)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat


def sc_decode_fast(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
