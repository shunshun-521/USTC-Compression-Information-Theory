"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    La 为上半分支，Lb 为下半分支
    """
    return Lb + (1.0 - 2.0 * u_hat) * La


def _active_llr_level(i, n):
    """译码索引 i 需要更新的 LLR 层起始层"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """译码索引 i 需要回传的比特层起始层"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        la, lb = llr_node[:half], llr_node[half:]
        decode_node(f_operation(la, lb), offset)
        decode_node(
            g_operation(la, lb, u_hat[offset:offset + half]),
            offset + half,
        )

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层 L/B 存储，按比特倒序索引顺序译码）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        _update_llrs(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].copy()


def _update_llrs(L, B, l, n):
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top = L[j, s]
                btm = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top, btm)
            else:
                btm = L[j, s]
                top = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top, btm, top_bit)


def _update_bits(B, l, n):
    N = B.shape[0]
    if l < N // 2:
        return

    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供参考/扩展）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (layer - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(bit_reversed(i, n), n), n)))
        if bit_reversed(i, n) >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(bit_reversed(i, n), n), -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec
