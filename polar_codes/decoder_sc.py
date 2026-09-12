"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SC）
"""
import math
import numpy as np

LLR_CLIP = 30.0


def clip_llr(llr):
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return clip_llr(
        np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    )


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    La 为上层（top），Lb 为下层（bottom）LLR。
    """
    u_hat = np.asarray(u_hat)
    return clip_llr((1 - 2 * u_hat) * La + Lb)


def _active_llr_level(i, n):
    """找到 i 的二进制展开中第一个 1 的位置（从高位起）。"""
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
    """找到 i 的二进制展开中第一个 0 的位置（从高位起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    g 运算使用当前阶段的中间码字比特（u_up），而非最终消息比特。
    """
    llr = clip_llr(np.asarray(llr, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_node[0]:
                bit = 0
            else:
                bit = 0 if llr_node[0] >= 0 else 1
            u_hat[idx] = bit
            return np.array([bit], dtype=int), np.array([bit], dtype=int)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        _, u_left_up = decode_node(llr_left, frozen_node[:half], bit_offset)

        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        _, u_right_up = decode_node(llr_right, frozen_node[half:], bit_offset + half)

        u_up = np.empty(n, dtype=int)
        u_up[:half] = u_left_up ^ u_right_up
        u_up[half:] = u_right_up
        return u_hat[bit_offset:bit_offset + n], u_up

    decode_node(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Permuted SC 风格）。
    """
    n = int(math.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []

    for l in decode_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)) if l >= N // 2 else [])

    lambda_offset = list(range(n + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted SC 译码主函数。
    输入信道 LLR 按码字自然顺序排列（与 polar_encode 输出一致）。
    """
    llr_ch = clip_llr(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
    llr_ch = llr_ch[br]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
