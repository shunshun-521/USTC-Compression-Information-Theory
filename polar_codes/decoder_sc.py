"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_block(llr_node, bit_offset, length):
        if length == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, bit_offset, half)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, bit_offset + half, half)

    decode_block(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        while (phi >> l) & 1:
            l += 1
        llr_layer_vec.append(list(range(l, n)))

        if phi == N - 1:
            bit_layer_vec.append(list(range(n)))
        else:
            l2 = 0
            while ((phi + 1) >> l2) & 1:
                l2 += 1
            bit_layer_vec.append(list(range(l2)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def sc_decode_sequential(llr_ch, frozen_bits):
    """按相位逐比特更新的 SC（P/C 数组调度，参考实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    lambda_offset, llr_layer_vec, bit_layer_vec = _SC_CACHE[N]

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        l = 0
        while (phi >> l) & 1:
            l += 1

        for layer in range(l, n):
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = lambda_offset[layer]
            La = P[layer + 1, pm : pm + span]
            Lb = P[layer + 1, pm + span : pm + 2 * span]
            P[layer, pm : pm + span] = f_operation(La, Lb)

        for layer in range(n - 1, l - 1, -1):
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = lambda_offset[layer]
            for beta in range(span):
                C[layer, pm + beta] = C[layer + 1, pm + beta]
                P[layer, pm + beta] = g_operation(
                    P[layer + 1, pm + beta],
                    P[layer + 1, pm + beta + span],
                    C[layer, pm + beta],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, 0] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1
            C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = lambda_offset[layer]
            for beta in range(span):
                C[layer + 1, pm + beta + span] = C[layer, pm + 2 * beta + 1]
                C[layer + 1, pm + beta] = (
                    C[layer, pm + 2 * beta] ^ C[layer, pm + 2 * beta + 1]
                )

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC：极化因子图上的单次双向消息传递（与 BP 单次迭代等价）。
    输入 LLR 需为译码器顺序（见 channel.channel_llr_to_decoder）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_idx = np.where(frozen_bits)[0]
    large = 1e6

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch
    L[frozen_idx, 0] = large

    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, s * 2):
            for k in range(s):
                idx_u = i + k
                idx_l = i + k + s
                L[idx_u, j - 1] = f_operation(R[idx_u, j] + L[idx_l, j], L[idx_u, j])
                L[idx_l, j - 1] = f_operation(R[idx_u, j], L[idx_u, j]) + L[idx_l, j]

    for j in range(0, n):
        s = 1 << j
        for i in range(0, N, s * 2):
            for k in range(s):
                idx_u = i + k
                idx_l = i + k + s
                R[idx_u, j + 1] = f_operation(R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j])
                R[idx_l, j + 1] = f_operation(R[idx_u, j], L[idx_u, j + 1]) + R[idx_l, j]

    u_hat = np.zeros(N, dtype=np.int8)
    for i in range(N):
        tot = L[i, 0] + R[i, 0]
        u_hat[i] = 0 if tot >= 0 else 1
    u_hat[frozen_idx] = 0
    return u_hat
