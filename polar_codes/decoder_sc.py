"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> Lb-La"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 dit.coding.polar 结构一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)
    decisions = [0] * n

    def sc_sub(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]
        half = len(indices) // 2
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        enc_left = sc_sub(left_llr, indices[:half])
        right_llr = np.array(
            [g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)],
            dtype=np.float64,
        )
        enc_right = sc_sub(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc_sub(llr, list(range(n)))
    return np.array(decisions, dtype=np.int8)


def precompute_sc_indices(N):
    """预计算非递归 SC 索引（简化占位，回退递归）"""
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC：当前回退到递归实现以保证正确性"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return sc_decode_recursive(llr_ch, frozen_bits)
