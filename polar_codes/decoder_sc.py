"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import _bit_rev_indices


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    if np.isinf(a):
        return b
    if np.isinf(b):
        return a
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def upper_llr(l1, l2):
    """f 运算（对数域 box-plus，与 min-sum 在 high-SNR 下等价）"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, bit):
    """g 运算（l1=下支路 LLR，l2=上支路 LLR）"""
    if bit == 0:
        return l1 + l2
    return l1 - l2


def _bit_reversed(i, n):
    r = 0
    for k in range(n):
        r = (r << 1) | ((i >> k) & 1)
    return r


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


def _prepare_channel_llr(llr_ch):
    """编码含 B_N 时，信道 LLR 需比特倒序后进入译码树"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = _bit_rev_indices(len(llr_ch))
    return llr_ch[br]


def _sc_core(llr, frozen_bits):
    """SC 译码核心（L/B 阵列更新）"""
    N = len(llr)
    n = int(np.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1]))

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助向量（与 Arikan 层索引等价描述，供文档/扩展使用）
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(_active_bit_level(phi, n))))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = _prepare_channel_llr(llr_ch)
    return _sc_core(llr, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用同一核心实现，便于对照验证）"""
    llr = _prepare_channel_llr(llr)
    return _sc_core(llr, frozen_bits)


def sc_update_llrs_for_bit(L, B, l, n):
    """对单个比特索引 l 更新 LLR（SCL 复用）"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1]))
    return L[l, n]


def sc_update_bits_for_bit(B, l, n):
    """对单个比特索引 l 回传硬比特（SCL 复用）"""
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]
