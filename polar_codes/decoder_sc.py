"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

try:
    from decoder_sc_fast import NUMBA_AVAILABLE, sc_decode_numba
except ImportError:
    NUMBA_AVAILABLE = False


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        bits = phi
        for layer in range(n):
            if (bits & 1) == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            bits >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_ch, frozen_bits):
    """SC 译码核心：全层 LLR 递推（正确性优先）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    L[n] = llr_ch[br]

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in range(n - 1, -1, -1):
            step = 1 << layer
            for i in range(0, N, 2 * step):
                for j in range(i, i + step):
                    L[layer, j] = f_operation(L[layer + 1, j], L[layer + 1, j + step])
                    L[layer, j + step] = g_operation(
                        L[layer + 1, j], L[layer + 1, j + step], C[layer, j]
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[0, phi] >= 0 else 1

        C[0, phi] = u_hat[phi]
        for layer in range(n):
            step = 1 << layer
            for i in range(0, N, 2 * step):
                for j in range(i, i + step):
                    C[layer + 1, j] = C[layer, j] ^ C[layer, j + step]
                    C[layer + 1, j + step] = C[layer, j + step]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    if NUMBA_AVAILABLE:
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        frozen_bits = np.asarray(frozen_bits, dtype=np.int64)
        br = bit_reversal_permutation(len(llr_ch))
        return sc_decode_numba(llr_ch, frozen_bits, br).astype(int)
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，保留接口兼容性）"""
    return _sc_decode_core(llr_ch, frozen_bits)
