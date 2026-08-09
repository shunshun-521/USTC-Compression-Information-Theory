"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _update_llr(L, B, x, n, f_fn=f_operation, g_fn=g_operation):
    N = L.shape[0]
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for i in range(x, N, s):
            if t > i % s:
                L[i, j] = f_fn(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_fn(
                    L[i - t, j + 1], L[i, j + 1], B[i - t, j]
                )


def _update_bits(B, x, n):
    N = B.shape[0]
    b = [x]
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        bnext = []
        for i in b:
            if t <= i % s:
                B[i - t, j + 1] = B[i, j] ^ B[i - t, j]
                B[i, j + 1] = B[i, j]
                bnext.extend([i, i - t])
        b = bnext


def _sc_decode_core(llr, frozen_bits):
    """非递归 SC 译码核心（LLR 已按比特倒序排列）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr
    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = _bit_reversed_index(i, n)
        _update_llr(L, B, l, n)
        if frozen_bits[l]:
            B[l, 0] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, 0] >= 0 else 1
            B[l, 0] = bit
            u_hat[l] = bit
        _update_bits(B, l, n)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    br = bit_reversal_permutation(len(llr))
    llr_br = np.asarray(llr, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_br)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_block(llrs, offset, depth):
        if depth == 0:
            i = offset
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llrs[0] >= 0 else 1
            return

        half_len = 1 << (depth - 1)
        llr_left = f_operation(llrs[:half_len], llrs[half_len:])
        decode_block(llr_left, offset, depth - 1)
        llr_right = g_operation(
            llrs[:half_len], llrs[half_len:], u_hat[offset:offset + half_len]
        )
        decode_block(llr_right, offset + half_len, depth - 1)

    decode_block(llr_br, 0, int(math.log2(N)))
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = []
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                layers = list(range(layer, n))
                break
        llr_layer_vec.append(layers)

        bit_layers = []
        tmp = phi
        layer = 0
        while tmp & 1:
            bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
        lambda_offset.append(phi)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 为码字自然顺序，内部做比特倒序后译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return _sc_decode_core(llr_ch[br], frozen_bits)
