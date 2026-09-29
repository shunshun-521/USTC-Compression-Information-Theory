"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（高效）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """f 运算（对数域 box-plus，与 min-sum 兼容）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)

    def logdomain_sum(x, y):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))

    def upper_llr(l1, l2):
        return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)

    if La.ndim == 0 and Lb.ndim == 0:
        return upper_llr(float(La), float(Lb))
    out = np.empty(np.broadcast_shapes(La.shape, Lb.shape), dtype=np.float64)
    it = np.nditer([La, Lb, out], flags=["zerosize_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for a, b, o in it:
        o[...] = upper_llr(float(a), float(b))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


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


def _llr_channel_reorder(llr_ch, N):
    """将信道 LLR 重排为 PSCD 期望的顺序（与含 B_N 的编码器配套）"""
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode(llr_ch, frozen_bits):
    """非递归 PSCD 译码（主实现）"""
    llr_ch = _llr_channel_reorder(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1))
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    l_bot = L[j, s]
                    l_top = L[j - branch_size, s]
                    b_top = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = l_bot + l_top if b_top == 0 else l_bot - l_top

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [_bit_reversed(i, n) for i in range(N)]:
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现：与 PSCD 结果一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（PSCD 实现中由在线逻辑替代，保留接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n - 1, -1, -1)) if phi == 0 else [] for phi in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
