"""Core SC tree decoder (exact box-plus f)."""
import numpy as np
import math


def f_boxplus(la, lb):
    la = np.asarray(la, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    t = np.tanh(la / 2.0) * np.tanh(lb / 2.0)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


def g_boxplus(la, lb, u):
    return (1 - 2 * np.asarray(u, dtype=np.int8)) * la + lb


def sc_tree_decode(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(L, depth, bit_pos):
        if depth == 0:
            if frozen_bits[bit_pos]:
                u_hat[bit_pos] = 0
            else:
                u_hat[bit_pos] = 0 if L[0] >= 0 else 1
            return
        half = 1 << (depth - 1)
        L_left = f_boxplus(L[:half], L[half:])
        decode_block(L_left, depth - 1, bit_pos)
        u_partial = u_hat[bit_pos : bit_pos + half]
        L_right = g_boxplus(L[:half], L[half:], u_partial)
        decode_block(L_right, depth - 1, bit_pos + half)

    decode_block(llr, n, 0)
    return u_hat
