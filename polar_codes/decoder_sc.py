"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（结构参考，大 N 较慢）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        layer = 0
        while p % 2 == 1:
            layer += 1
            p >>= 1
        bit_layer_vec.append(list(range(layer)))
    return lambda_offset, bit_layer_vec


_SC_PRECOMPUTE = {}


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    每个相位对需更新的 LLR 层做整块 f/g 递推，保证数值正确性。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    if N not in _SC_PRECOMPUTE:
        _SC_PRECOMPUTE[N] = precompute_sc_indices(N)
    bit_layer_vec = _SC_PRECOMPUTE[N][1]

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        p = phi
        layer = 0
        while p % 2 == 1:
            layer += 1
            p >>= 1

        for l in range(n - 1, layer - 1, -1):
            stride = 1 << l
            num_blocks = N // (2 * stride)
            pl = P[l]
            pl1 = P[l + 1]
            cl = C[l]
            for block in range(num_blocks):
                base = block * 2 * stride
                for j in range(stride):
                    idx = base + j
                    pl[idx] = f_operation(pl1[idx], pl1[idx + stride])
                    pl[idx + stride] = g_operation(
                        pl1[idx], pl1[idx + stride], cl[idx]
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1

        C[0, phi] = u_hat[phi]

        for l in bit_layer_vec[phi]:
            stride = 1 << l
            num_blocks = N // (2 * stride)
            cl = C[l]
            cl1 = C[l + 1]
            for block in range(num_blocks):
                base = block * 2 * stride
                for j in range(stride):
                    idx = base + j
                    cl1[idx] = cl[idx] ^ cl[idx + stride]
                    cl1[idx + stride] = cl[idx + stride]

    return u_hat
