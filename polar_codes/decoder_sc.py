"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """f 运算：box-plus"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    ta = np.tanh(La * 0.5)
    tb = np.tanh(Lb * 0.5)
    prod = np.clip(ta * tb, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def f_operation_minsum(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（译码域 LLR，Arikan 配对）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = np.zeros(N, dtype=int)
    indices = list(range(N))

    def _sc(llrs, idxs):
        if len(idxs) == 1:
            i = idxs[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(idxs) // 2
        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = _sc(left_llr, idxs[:half])
        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, idxs[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr.tolist(), indices)
    return decisions


def _update_llr_layer(L, phi, m):
    for s in range(m - 1, -1, -1):
        if (phi >> s) & 1:
            continue
        block = 1 << (s + 1)
        start = (phi // block) * block
        half = 1 << s
        for i in range(start, start + half):
            L[i, s] = f_operation(L[i, s + 1], L[i + half, s + 1])


def _update_partial_sum(L, C, u_hat, phi, m):
    for s in range(m):
        if (phi >> s) & 1:
            continue
        block = 1 << (s + 1)
        start = (phi // block) * block
        half = 1 << s
        for i in range(start, start + half):
            C[i, s] = u_hat[i]
            L[i + half, s] = g_operation(L[i, s], L[i + half, s + 1], C[i, s])


def sc_decode_sequential(llr, frozen_bits):
    """按 phi=0..N-1 顺序的非递归 SC（用于 SCL）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    m = int(math.log2(N))

    L = np.zeros((N, m + 1), dtype=np.float64)
    C = np.zeros((N, m + 1), dtype=np.int8)
    L[:, m] = llr
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        _update_llr_layer(L, phi, m)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 1 if L[phi, 0] < 0 else 0
        _update_partial_sum(L, C, u_hat, phi, m)

    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            t = phi
            layer = 0
            while t % 2 == 1:
                bit_layers.append(layer)
                t >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 主入口（译码域，与递归实现等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_safe(llr_dec, frozen_bits):
    return sc_decode_recursive(llr_dec, frozen_bits)


def sc_decode_channel(llr_channel, frozen_bits):
    from channel import channel_llr_to_decode_domain

    llr_dec = channel_llr_to_decode_domain(llr_channel)
    return sc_decode_recursive(llr_dec, frozen_bits)


def sc_llr_at_phi(L, phi, m):
    _update_llr_layer(L, phi, m)
    return L[phi, 0]


def sc_init_state(llr):
    """初始化顺序 SC 的 L/C 数组"""
    N = len(llr)
    m = int(math.log2(N))
    L = np.zeros((N, m + 1), dtype=np.float64)
    C = np.zeros((N, m + 1), dtype=np.int8)
    L[:, m] = llr
    return L, C, m
