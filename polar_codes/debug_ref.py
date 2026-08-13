"""Direct port of polar-codes SCD for debugging."""
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def f_ms(a, b):
    sa = 1 if a >= 0 else -1
    sb = 1 if b >= 0 else -1
    return sa * sb * min(abs(a), abs(b))


def g_ms(a, b, u):
    return a + b if u == 0 else a - b


def polar_encode(u):
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def sc_decode_ref(llr_ch, frozen_set):
    N = len(llr_ch)
    n = int(np.log2(N))
    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1))
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed(phi, n)
        for s in range(n - active_llr_level(l, n), n):
            bs = 1 << (s + 1)
            br = bs // 2
            for j in range(l, N, bs):
                if j % bs < br:
                    L[j, s + 1] = f_ms(L[j, s], L[j + br, s])
                else:
                    L[j, s + 1] = g_ms(L[j - br, s], L[j, s], int(B[j - br, s + 1]))

        if phi in frozen_set:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[l, n] >= 0 else 1
        B[l, n] = u_hat[phi]

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                bs = 1 << s
                br = bs // 2
                for j in range(l, -1, -bs):
                    if j % bs >= br:
                        B[j - br, s - 1] = int(B[j, s]) ^ int(B[j - br, s])
                        B[j, s - 1] = B[j, s]
    return u_hat


u = np.array([1, 0, 1, 1])
x = polar_encode(u)
print('u', u, 'x', x)
llr = (1 - 2 * x) * 100.0
uh = sc_decode_ref(llr, set())
print('uh', uh)

from decoder_sc import sc_decode

uh2 = sc_decode(llr, np.zeros(4, dtype=int))
print('my', uh2)