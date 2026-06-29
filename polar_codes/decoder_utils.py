"""SC/SCL 译码辅助函数（参考 Permuted SCD 实现）。"""
import numpy as np


def bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def hard_decision(y):
    return 0 if y >= 0 else 1


def f_minsum(l1, l2, alpha=1.0):
    """min-sum 近似 f 运算。"""
    return alpha * np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))


def g_llr(l1, l2, b):
    return (l1 + l2) if b == 0 else (l1 - l2)


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
