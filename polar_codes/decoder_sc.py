"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（upper_llr）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（lower_llr）：La 为上层 LLR，Lb 为下层 LLR
    g(La, Lb, u_hat) = La + Lb (u=0) 或 Lb - La (u=1)
    等价于 (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr_exact(l1, l2):
    """精确 log-domain box-plus（f 运算）"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(btm_llr, top_llr, top_bit):
    """g 运算：参数顺序为 (下层, 上层, 上层已译比特)"""
    if top_bit == 0:
        return btm_llr + top_llr
    return btm_llr - top_llr


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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，使用精确 box-plus）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr = llr[br]
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
        top = llr_node[:half]
        btm = llr_node[half:]
        llr_left = np.array([upper_llr_exact(top[i], btm[i]) for i in range(half)])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = np.array([lower_llr(btm[i], top[i], u_left[i]) for i in range(half)])
        decode_node(llr_right, bit_offset + half)

    import math
    if 2 ** int(math.log2(N)) != N:
        raise ValueError("N must be a power of 2")
    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助信息"""
    n = int(np.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return n, decode_order


def sc_decode_core(llr_ch, frozen_bits, use_min_sum=False):
    """
    非递归 SC 译码核心（输入 LLR 已按比特倒序排列）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n, decode_order = precompute_sc_indices(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    f_func = f_operation if use_min_sum else upper_llr_exact

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_func(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for l in decode_order:
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits, use_min_sum=False):
    """
    非递归 SC 译码（高效实现，按比特倒序相位译码）。
    信道 LLR 为自然顺序（对应编码输出顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return sc_decode_core(llr_ch[br], frozen_bits, use_min_sum=use_min_sum)
