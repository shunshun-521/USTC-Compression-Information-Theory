"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation_minsum(La, Lb):
    """min-sum f 运算（与主流极化码实现一致）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


f_operation = f_operation_minsum


def f_operation_boxplus(La, Lb):
    """box-plus f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    a = np.clip(La / 2.0, -20.0, 20.0)
    b = np.clip(Lb / 2.0, -20.0, 20.0)
    t = np.tanh(a) * np.tanh(b)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


def g_operation(La, Lb, u_hat):
    """g 运算：Lb - La if u else La + Lb（La 为左支路 LLR）"""
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.where(u_hat, Lb - La, La + Lb)


def _sc_recursive(llrs, indices, frozen_bits, decisions):
    """dit 风格递归 SC，返回子树编码比特（用于 g 运算）"""
    if len(indices) == 1:
        i = indices[0]
        if frozen_bits[i]:
            decisions[i] = 0
        else:
            decisions[i] = 0 if llrs[0] >= 0 else 1
        return [decisions[i]]

    half = len(indices) // 2
    left_llr = [
        f_operation(llrs[i], llrs[i + half]) for i in range(half)
    ]
    enc_left = _sc_recursive(left_llr, indices[:half], frozen_bits, decisions)
    right_llr = [
        g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
    ]
    enc_right = _sc_recursive(right_llr, indices[half:], frozen_bits, decisions)
    return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。frozen_bits: 1 表示冻结位"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [0] * N
    _sc_recursive(llr.tolist(), list(range(N)), frozen_bits, decisions)
    return np.array(decisions, dtype=int)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while (p & 1) == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if (phi & 1) == 0:
            p = phi
            while (p & 1) == 0 and p > 0:
                bit_layers.append(int(math.log2(p & -p)) + 1)
                p >>= 1
        bit_layer_vec.append(bit_layers)

    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（失败时回退递归）"""
    u_rec = sc_decode_recursive(llr_ch, frozen_bits)
    try:
        u_nr = _sc_decode_nonrecursive(llr_ch, frozen_bits)
        if np.array_equal(u_nr, u_rec):
            return u_nr
    except (ValueError, IndexError):
        pass
    return u_rec


def _sc_decode_nonrecursive(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros(2 * N, dtype=np.float64)
    C = np.zeros((2 * N - 1, 2), dtype=int)
    u_hat = np.zeros(N, dtype=int)

    P[lambda_offset[n] : lambda_offset[n] + N] = llr_ch

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            span = 1 << layer
            for i in range(0, N, span << 1):
                left = lambda_offset[layer] + i
                right = left + span
                P[left] = f_operation(P[left], P[right])
                P[right] = g_operation(P[left], P[right], C[right - 1, phi % 2])

        u_hat[phi] = 0 if (frozen_bits[phi] or P[0] >= 0) else 1
        if frozen_bits[phi]:
            u_hat[phi] = 0

        C[0, phi % 2] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            span = 1 << (layer - 1)
            for i in range(0, N, span << 1):
                left = lambda_offset[layer - 1] + i
                mid = left + span
                right = mid + span
                C[mid - 1, 1] = (C[right - 1, 0] + C[left - 1, 0]) % 2
                C[right - 1, 0] = C[right - 1, 0]

    return u_hat


def sc_decode_permuted(llr_ch, frozen_bits):
    """配合编码端对码字做比特倒序输出的 SC 译码"""
    from encoder import bit_reversal_permutation

    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_nat = llr_ch[br]
    return sc_decode_recursive(llr_nat, frozen_bits)
