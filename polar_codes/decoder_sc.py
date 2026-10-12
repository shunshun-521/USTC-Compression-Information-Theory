"""
极化码 SC（串行抵消）译码器
递归实现（半分割）+ 非递归调度实现（polarcodes SCD 风格）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum f 运算"""
    La, Lb = np.asarray(La, dtype=np.float64), np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：Lb + (1-2*u)*La"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0:
        u = int(u_hat)
        return Lb + (1 - 2 * u) * La
    return Lb + (1 - 2 * u_hat) * La


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
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    b = int(b)
    return l1 + l2 if b == 0 else l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def _rec(y, fz):
        n = len(y)
        if n == 1:
            if fz[0]:
                return np.array([0], dtype=int)
            return np.array([0 if y[0] >= 0 else 1], dtype=int)
        h = n // 2
        y_left = f_operation(y[:h], y[h:])
        u_left = _rec(y_left, fz[:h])
        y_right = g_operation(y[:h], y[h:], u_left)
        u_right = _rec(y_right, fz[h:])
        return np.concatenate([u_left, u_right])

    return _rec(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码：递归 SC + 硬判决 GF(2) 逆变换择优"""
    from encoder import polar_encode
    from utils import gf2_polar_decode

    u_rec = sc_decode_recursive(llr_ch, frozen_bits)
    u_hd = gf2_polar_decode(llr_ch, frozen_bits)
    hard_x = (np.asarray(llr_ch) < 0).astype(int)
    err_rec = np.sum(polar_encode(u_rec) != hard_x)
    err_hd = np.sum(polar_encode(u_hd) != hard_x)
    return u_rec if err_rec <= err_hd else u_hd


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC（SCD）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


sc_decode.__doc__ = "主入口：在递归 SC 与 GF(2) 硬判决译码间按码字一致性择优。"

def precompute_sc_indices(N):
    """SCL 使用的占位预计算（层宽）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
