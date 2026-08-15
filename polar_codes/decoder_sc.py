"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 的 min-sum 近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """精确 box-plus 运算（用于递归参考实现）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s = np.sign(La) * np.sign(Lb)
    a, b = np.abs(La), np.abs(Lb)
    minab = np.minimum(a, b)
    maxab = np.maximum(a, b)
    return s * (minab - np.log1p(np.exp(minab - maxab)))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找到 i 的二进制表示中第一个 1 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """找到 i 的二进制表示中第一个 0 的位置（从高位计）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n):
    """更新第 l 个比特对应的 LLR 树。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_boxplus(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n, N):
    """比特回传更新。"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，使用精确 box-plus）。
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_block(llr_blk, offset, size):
        if size == 1:
            i = offset
            u_hat[i] = 0 if frozen_bits[i] or llr_blk[0] >= 0 else 1
            return
        half = size // 2
        llr_left = f_boxplus(llr_blk[:half], llr_blk[half:])
        decode_block(llr_left, offset, half)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_blk[:half], llr_blk[half:], u_left)
        decode_block(llr_right, offset + half, half)

    decode_block(np.asarray(llr, dtype=np.float64), 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** layer - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        layers = []
        for layer in range(n):
            if phi_bin[n - 1 - layer] == "0":
                layers.append(layer)
        llr_layer_vec.append(layers)

        layers_bit = []
        if phi % 2 == 1:
            for layer in range(n):
                if phi_bin[n - 1 - layer] == "1":
                    layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（置换 SC 算法，高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    def bit_reversed(i):
        result = 0
        for bit in range(n):
            if i & (1 << bit):
                result |= 1 << (n - 1 - bit)
        return result

    for i in range(N):
        l = bit_reversed(i)
        _update_llrs(L, B, l, n)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
