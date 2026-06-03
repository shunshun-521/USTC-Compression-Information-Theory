"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归版本（逐层 LLR 更新，与标准 PSC 译码等价）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """f 运算（boxplus 的 min-sum 近似，用于对照）"""
    if np.isinf(La) and not np.isinf(Lb):
        return Lb
    if np.isinf(Lb) and not np.isinf(La):
        return La
    if np.isinf(La) and np.isinf(Lb):
        return np.inf
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 上支路，Lb 下支路"""
    if u_hat == 0:
        if np.isinf(La) or np.isinf(Lb):
            return np.inf
        return La + Lb
    return La - Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2**s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def channel_llr_to_decoder(llr_ch, N):
    """
    将信道 LLR 重排为译码树顺序。
    编码含比特倒序 x[i]=v[br(i)] 时，译码器输入需 v[j] 的 LLR，即 llr_ch[br^{-1}(j)]。
    """
    perm = bit_reversal_permutation(N)
    inv_perm = np.argsort(perm)
    return np.asarray(llr_ch, dtype=np.float64)[inv_perm]


def sc_decode(llr_ch, frozen_bits, apply_br_reorder=True):
    """
    非递归 SC 译码。
    frozen_bits[i]=True 表示冻结位（译码为 0）。
    apply_br_reorder: 编码器使用比特倒序时设为 True。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    if apply_br_reorder:
        llr_work = channel_llr_to_decoder(llr_ch, N)
    else:
        llr_work = llr_ch.copy()

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_work

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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（前半/后半拆分，用于对照）"""
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
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算 SC 辅助索引"""
    n = int(np.log2(N))
    lambda_offset = np.array([(1 << (n - layer + 1)) - 1 for layer in range(n + 1)])
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(
            list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        )
    return lambda_offset, llr_layer_vec, bit_layer_vec
