"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 PSC 实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_mask(frozen_bits):
    """统一冻结位表示：True/1 表示冻结位"""
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def bit_reversed(i, n):
    """比特倒序索引"""
    return int(format(i, f'0{n}b')[::-1], 2)


def active_llr_level(i, n):
    """PSC：从 MSB 起第一个 0 的位置（参考 polarcodes 库）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """PSC：从 MSB 起第一个 1 的位置"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=int)
        for i in range(half):
            u_left[i] = decode_node(llr_left[i:i + 1], depth - 1, bit_offset + i)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


# ==================== 非递归 SC 译码（PSC 高效实现）====================


def precompute_sc_indices(N):
    """
    预计算 PSC 译码辅助向量。
    返回 decode_order, llr_layer_ranges, bit_layer_ranges
    """
    n = int(math.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        start_b = n - active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, start_b - 1, -1)))
    return decode_order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSC SC 译码。
    信道 LLR 对应编码器输出顺序（含比特倒序置换），内部自动对齐。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    # 编码器输出含比特倒序，PSC 译码器期望蝶形域顺序
    rev = np.array([bit_reversed(i, n) for i in range(N)])
    llr_aligned = llr_ch[rev]

    decode_order, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_aligned.copy()
    u_hat = np.zeros(N, dtype=int)

    for step, l in enumerate(decode_order):
        for s in llr_layer_vec[step]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in bit_layer_vec[step]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
