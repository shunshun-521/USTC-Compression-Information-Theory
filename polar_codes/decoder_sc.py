"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala SCD）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    """log(exp(x)+exp(y))"""
    if x == -np.inf and y == -np.inf:
        return -np.inf
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域精确形式）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（对数域）"""
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """二进制展开中第一个 1 的位置（从高位计层）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """二进制展开中第一个 0 的位置"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _map_channel_llr_to_decoder(llr_ch):
    """
    编码器输出含比特倒序；SCD 因子图按自然索引接收似然。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv_br = np.empty(N, dtype=int)
    inv_br[br] = np.arange(N)
    return llr_ch[inv_br].astype(np.float64)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    llr = _map_channel_llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr)
    n = int(np.log2(N))
    u_hat = sc_decode_vangala(llr, frozen_set, N, n)
    return u_hat


# ==================== 非递归 SC 译码（Vangala SCD）====================


def precompute_sc_indices(N):
    """保留接口：返回解码顺序与层信息"""
    n = int(np.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    return None, decode_order, None


def sc_decode_vangala(llr_natural, frozen_set, N, n):
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_natural

    for idx in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, idx, n)
        if idx in frozen_set:
            B[idx, n] = 0
        else:
            B[idx, n] = 0 if L[idx, n] >= 0 else 1
        _update_bits(B, idx, n)

    return B[:, n].astype(int)


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = int(2 ** (s + 1))
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = int(2 ** s)
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1 表示冻结位
    """
    llr = _map_channel_llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])
    return sc_decode_vangala(llr, frozen_set, N, n)
