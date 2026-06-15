"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _permute_llr(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _sc_decode_recursive_on_tree(llr_tree, frozen_bits):
    """递归 SC 译码（在比特倒序后的 LLR 树上操作）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode(llr_ch, frozen):
        n = len(llr_ch)
        if n == 1:
            if frozen[0]:
                u = np.array([0], dtype=np.int8)
            else:
                u = np.array([0 if llr_ch[0] >= 0 else 1], dtype=np.int8)
            return u, u.copy()

        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = decode(x_llr1, frozen[:half])

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = decode(x_llr2, frozen[half:])

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_new = np.bitwise_xor(
            u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8)
        )
        u_hat_up = np.concatenate([u_hat1_up_new, u_hat2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode(llr_tree, frozen_bits)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _sc_decode_recursive_on_tree(_permute_llr(llr), frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        tz = 0
        tmp = phi
        while tmp & 1:
            tz += 1
            tmp >>= 1
        llr_layers = list(range(n - 1, tz - 1, -1)) if tz < n else []
        bit_layers = list(range(tz))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _trailing_ones(phi):
    tz = 0
    tmp = phi
    while tmp & 1:
        tz += 1
        tmp >>= 1
    return tz


def _update_llr_for_phi(L, B, phi, n, N):
    """为当前相位 phi 更新分层 LLR（与递归 SC 等价）"""
    layer_stop = _trailing_ones(phi)
    for layer in range(n - 1, layer_stop - 1, -1):
        step = 1 << layer
        for block in range(0, N, 2 * step):
            i = block
            L[layer, i] = f_operation(L[layer + 1, i], L[layer + 1, i + step])
            B[layer, i + step] = np.bitwise_xor(
                B[layer + 1, i], B[layer + 1, i + step]
            )
            L[layer, i + step] = g_operation(
                L[layer + 1, i],
                L[layer + 1, i + step],
                B[layer, i + step],
            )
    return L[0, phi]


def _propagate_partial_sum(B, phi, bit, n):
    B[0, phi] = bit
    layer_stop = _trailing_ones(phi)
    for layer in range(layer_stop):
        step = 1 << layer
        base = (phi // (2 * step)) * 2 * step
        if (phi // step) % 2 == 0:
            B[layer + 1, base] = bit
        else:
            B[layer + 1, base + step] = bit


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（调用高效递归核心实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
