"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

# ==================== 基本运算 ====================


def _sign_pm(x):
    return 1.0 if x >= 0 else -1.0


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（标量或数组）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _f_scalar(a, b):
    return _sign_pm(a) * _sign_pm(b) * min(abs(a), abs(b))


def _g_scalar(a, b, u):
    return b - a if u else b + a


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def sc(llr_node, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_node[0] >= 0 else 1
            return [u_hat[i]]

        half = len(indices) // 2
        left_llr = [
            _f_scalar(llr_node[i], llr_node[i + half]) for i in range(half)
        ]
        enc_left = sc(left_llr, indices[:half])
        right_llr = [
            _g_scalar(llr_node[i], llr_node[i + half], enc_left[i])
            for i in range(half)
        ]
        enc_right = sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc(llr.tolist(), list(range(N)))
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

_SC_SCHEDULE = {}


def precompute_sc_indices(N):
    """
    预计算译码阶段信息：自然比特顺序下的 LLR/比特回传层索引。
    """
    if N in _SC_SCHEDULE:
        return _SC_SCHEDULE[N]

    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for l in range(N):
        llr_layers = []
        p = l
        layer = 0
        while (p & 1) == 1:
            llr_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layers = []
        p = l + 1
        layer = 0
        while (p & 1) == 1:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    result = (llr_layer_vec, bit_layer_vec, n)
    _SC_SCHEDULE[N] = result
    return result


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：显式栈模拟递归，与 sc_decode_recursive 等价。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    op_stack = [(llr_ch.tolist(), list(range(N)), 0, None)]
    partial_stack = []

    while op_stack:
        llr_node, indices, phase, saved_enc_left = op_stack.pop()

        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_node[0] >= 0 else 1
            partial_stack.append([u_hat[i]])
            continue

        half = len(indices) // 2
        if phase == 0:
            left_llr = [
                _f_scalar(llr_node[i], llr_node[i + half]) for i in range(half)
            ]
            op_stack.append((llr_node, indices, 1, None))
            op_stack.append((left_llr, indices[:half], 0, None))
        elif phase == 1:
            enc_left = partial_stack.pop()
            right_llr = [
                _g_scalar(llr_node[i], llr_node[i + half], enc_left[i])
                for i in range(half)
            ]
            op_stack.append((llr_node, indices, 2, enc_left))
            op_stack.append((right_llr, indices[half:], 0, None))
        else:
            enc_right = partial_stack.pop()
            enc_left = saved_enc_left
            partial_stack.append(
                [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right
            )

    return u_hat
