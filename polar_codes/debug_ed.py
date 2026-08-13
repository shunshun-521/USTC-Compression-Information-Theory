"""Debug encoder/decoder consistency."""
import numpy as np
from construction import ga_construction


def F(n):
    if n == 1:
        return np.array([[1, 0], [1, 1]], dtype=int)
    Fm = F(n - 1)
    Z = np.zeros_like(Fm)
    return np.block([[Fm, Z], [Fm, Fm]])


def bitrev_perm(N):
    m = int(np.log2(N))
    return np.array([int(format(i, f"0{m}b")[::-1], 2) for i in range(N)])


def polar_encode_matrix(u):
    N = len(u)
    G = (np.eye(N, dtype=int)[bitrev_perm(N)] @ F(int(np.log2(N)))) % 2
    return (np.array(u, dtype=int) @ G) % 2


def fop(a, b):
    sa = np.sign(a)
    sb = np.sign(b)
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    return sa * sb * np.minimum(np.abs(a), np.abs(b))


def gop(a, b, u):
    return (1 - 2 * u) * a + b


def sc_rec(llr, frozen):
    N = len(llr)
    uhat = np.zeros(N, dtype=int)

    def dec(node, off):
        n = len(node)
        if n == 1:
            i = off
            uhat[i] = 0 if frozen[i] or node[0] >= 0 else 1
            return
        h = n // 2
        dec(fop(node[:h], node[h:]), off)
        dec(gop(node[:h], node[h:], uhat[off : off + h]), off + h)

    dec(np.array(llr, float), 0)
    return uhat


for N in [4, 8, 16, 32, 64]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode_matrix(u)
        llr = (1 - 2 * x) * 50.0
        uh = sc_rec(llr, frozen.astype(bool))
        if np.array_equal(uh, u):
            ok += 1
    print(f"N={N}: matrix encode + simple SC: {ok}/100")
