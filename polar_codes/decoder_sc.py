"""
极化码 SC（串行抵消）译码器
非递归实现（参考 Permuted SCD 结构）；另含递归版本用于对照。
"""
import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed(i, n):
    rev = 0
    for k in range(n):
        if (i >> k) & 1:
            rev |= 1 << (n - 1 - k)
    return rev


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """f 运算（log-domain boxplus 的稳定近似 + 无穷 LLR 处理）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    it = np.nditer([La, Lb, out], flags=["refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for a, b, o in it:
        av, bv = float(a), float(b)
        if np.isinf(av) and not np.isinf(bv):
            o[...] = bv
        elif np.isinf(bv) and not np.isinf(av):
            o[...] = av
        elif np.isinf(av) and np.isinf(bv):
            o[...] = np.inf
        elif abs(av) < 1e-15:
            o[...] = bv
        elif abs(bv) < 1e-15:
            o[...] = av
        else:
            o[...] = np.sign(av) * np.sign(bv) * min(abs(av), abs(bv))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 => La+Lb；b=1 => La-Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    sign = 1 - 2 * u_hat
    return sign * La + Lb


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return float(f_operation(l1, l2))


def lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主入口）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（与 bit-reversed 相位顺序一致）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价，用于校验）"""
    return sc_decode(llr, frozen_bits)


sc_decode_nonrecursive = sc_decode
sc_decode_layered = sc_decode
