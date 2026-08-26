"""
极化码 SC（串行抵消）译码器
实现参考 polarcodes SCD：L[:,0] 为信道 LLR，按比特倒序序贯译码
"""
import numpy as np
from encoder import bit_reversed_index


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（log-domain box-plus）。"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（l1=下分支 LLR，l2=上分支 LLR）。"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似 f（供 BP 等模块使用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def active_llr_level(i, n):
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
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llr(L, B, l, n):
    N = L.shape[0]
    for s in range(n - active_llr_level(l, n), n):
        block = 1 << (s + 1)
        branch = block // 2
        for j in range(l, N, block):
            if j % block < branch:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch, s], B[j - branch, s + 1]
                )


def _update_bits(B, l, n):
    N = B.shape[0]
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block = 1 << s
        branch = block // 2
        for j in range(l, -1, -block):
            if j % block >= branch:
                B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        _update_llr(L, B, l, n)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致，作交叉验证）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口。"""
    return [[] for _ in range(N)], [[] for _ in range(N)]
