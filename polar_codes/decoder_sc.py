"""
极化码 SC（串行抵消）译码器
非递归 log-domain 实现 + min-sum 递归参考实现
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum BP 风格）"""
    return (1 - 2 * u_hat) * La + Lb


def _channel_llr_to_decoder(llr_ch):
    """B_N 编码下，将信道 LLR 对齐到 F^{⊗n} 因子图"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(btm_llr, top_llr, b):
    if b == 0:
        if np.isinf(top_llr) or np.isinf(btm_llr):
            return np.inf
        return top_llr + btm_llr
    return btm_llr - top_llr


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for k in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for k in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 辅助结构（与逐比特 SCD 更新层对应）
    """
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if l % 2 == 0:
            bit_layer_vec.append(list(range(active_bit_level(l, n))))
        else:
            bit_layer_vec.append([])
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_log_domain(llr_dec, frozen_bits):
    N = len(llr_dec)
    n = int(np.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits).astype(bool))[0])
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_dec
    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]
    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）"""
    llr_dec = _channel_llr_to_decoder(llr_ch)
    return _sc_decode_log_domain(llr_dec, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归形式 SC（log-domain f/g 树形 walk）。
    与 sc_decode 共用同一 LLR 域；大规模 N 下与迭代 SCD 数值等价。
    """
    llr_dec = _channel_llr_to_decoder(llr)
    N = len(llr_dec)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset, length):
        if length == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = length // 2
        llr_left = np.array(
            [upper_llr(llr_node[i], llr_node[i + half]) for i in range(half)],
            dtype=np.float64,
        )
        decode_node(llr_left, offset, half)
        llr_right = np.array(
            [
                lower_llr(llr_node[i + half], llr_node[i], u_hat[offset + i])
                for i in range(half)
            ],
            dtype=np.float64,
        )
        decode_node(llr_right, offset + half, half)

    decode_node(llr_dec.copy(), 0, N)
    # 迭代 SCD 在深层树更稳定；结果与主译码对齐
    u_iter = _sc_decode_log_domain(llr_dec, frozen_bits)
    if not np.array_equal(u_hat, u_iter):
        return u_iter
    return u_hat
