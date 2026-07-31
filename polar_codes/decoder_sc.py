"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _safe_bit(b):
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return 0
    return int(b)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    if np.ndim(u_hat) == 0 and np.isnan(u_hat):
        u_hat = 0
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _active_llr_level(i, n):
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
    mask = 2 ** (n - 1)
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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [(1 << layer) - 1 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))

        if l < N / 2:
            bit_layer_vec.append([])
        else:
            end_s = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end_s, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def _sc_decode_permuted(llr_ch, frozen_bits):
    """
    基于置换 SC 的分层译码（与极化码标准蝶形图一致）。
  L[j,0] 为信道 LLR，判决在 L[l,n]。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
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

        if i in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = _safe_bit(B[j, s]) ^ _safe_bit(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（扁平数组存储）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec, decode_order = precompute_sc_indices(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for idx, l in enumerate(decode_order):
        i = idx  # 自然顺序比特索引
        for s in llr_layer_vec[idx]:
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if i in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        for s in bit_layer_vec[idx]:
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = _safe_bit(B[j, s]) ^ _safe_bit(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def _permute_channel_llr(llr_ch, N):
    """将信道顺序 LLR 置换为 SC 蝶形图所需顺序"""
    n = int(math.log2(N))
    br = [_bit_reversed(i, n) for i in range(N)]
    llr_perm = np.empty(N, dtype=np.float64)
    llr_perm[br] = llr_ch
    return llr_perm


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入 llr_ch 为信道传输顺序（与 polar_encode 输出一致），
    返回自然顺序的 u_hat。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = [_bit_reversed(i, n) for i in range(N)]
    B_out = _sc_decode_permuted(llr_ch, frozen_bits)
    return B_out[br]


def sc_decode_recursive_wrapped(llr_ch, frozen_bits):
    """递归 SC 译码（自动 LLR 置换）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr_perm = _permute_channel_llr(llr_ch, N)
    return sc_decode_recursive(llr_perm, frozen_bits)
