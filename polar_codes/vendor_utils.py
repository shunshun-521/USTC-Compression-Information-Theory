"""Minimal utils for vendored SCD (from mcba1n/polar-codes)."""
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    return x + np.log1p(np.exp(y - x)) if x > y else y + np.log1p(np.exp(x - y))
