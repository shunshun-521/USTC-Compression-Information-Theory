"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与 polarcodes.SCD 同结构）
"""
import math
import numpy as np


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算：精确 log-domain boxplus（大 LLR 时退化为 min-sum）。 """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if np.isscalar(La) and np.isscalar(Lb):
        if abs(La) > 30 and abs(Lb) > 30:
            return float(np.sign(La) * np.sign(Lb) * min(abs(La), abs(Lb)))
        return float(_logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb))
    out = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    la = np.broadcast_to(La, out.shape).copy()
    lb = np.broadcast_to(Lb, out.shape).copy()
    mask = (np.abs(la) > 30) & (np.abs(lb) > 30)
    out[mask] = np.sign(la[mask]) * np.sign(lb[mask]) * np.minimum(
        np.abs(la[mask]), np.abs(lb[mask])
    )
    nm = ~mask
    for idx in np.ndindex(out.shape):
        if not nm[idx]:
            continue
        a, b = la[idx], lb[idx]
        out[idx] = _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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
    """递归 SC（子树索引 + strided f/g）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def _sc(llrs, indices):
        m = len(indices)
        if m == 1:
            i = indices[0]
            u_hat[i] = 0 if frozen_bits[i] or llrs[0] >= 0 else 1
            if frozen_bits[i]:
                u_hat[i] = 0
            return
        half = m // 2
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        _sc(left_llr, indices[:half])
        enc_left = u_hat[indices[:half]]
        right_llr = np.array(
            [
                g_operation(llrs[i], llrs[i + half], enc_left[i])
                for i in range(half)
            ],
            dtype=np.float64,
        )
        _sc(right_llr, indices[half:])

    _sc(llr.copy(), list(range(N)))
    return u_hat


def precompute_sc_indices(N):
    """返回译码相位顺序及层索引（供 SCL 复用）。"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（按比特倒序相位更新 LLR 树）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

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
        u_hat[l] = B[l, n]
        update_bits(l)

    return u_hat
