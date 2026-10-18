"""极化码工具（SCD 预计算与对数域运算）"""
import numpy as np


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


def upper_llr_boxplus(l1, l2):
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def upper_llr_minsum(l1, l2):
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    s1 = np.sign(l1)
    s2 = np.sign(l2)
    s1 = np.where(s1 == 0, 1.0, s1)
    s2 = np.where(s2 == 0, 1.0, s2)
    return s1 * s2 * np.minimum(np.abs(l1), np.abs(l2))


def lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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
