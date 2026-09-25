"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（PDF 算法 3–5，高效实现）
"""
import numpy as np
from encoder import bit_reversed


def logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.clip(x, -500, 500)
    y = np.clip(y, -500, 500)
    return np.where(
        x > y,
        x + np.log1p(np.exp(np.clip(y - x, -500, 500))),
        y + np.log1p(np.exp(np.clip(x - y, -500, 500))),
    )


def f_operation(La, Lb):
    """精确 log-domain f 运算（box-plus），支持向量化"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算，支持向量化；未译码位（nan）返回 nan"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    result = np.full(np.broadcast(La, Lb, u_hat).shape, np.nan, dtype=np.float64)
    mask0 = u_hat == 0
    mask1 = u_hat == 1
    result = np.where(mask0, La + Lb, result)
    result = np.where(mask1, La - Lb, result)
    return result


def f_min_sum(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs_pdf(L, B, x, n):
    """Algorithm 4: channel LLR 在列 n，判决在列 0"""
    for j in range(n - 1, -1, -1):
        block = 1 << (n - j)
        half = block // 2
        for i in range(x, L.shape[0], block):
            if half > i % block:
                L[i, j] = f_operation(L[i, j + 1], L[i + half, j + 1])
            else:
                top_bit = B[i - half, j]
                if not np.isnan(top_bit):
                    L[i, j] = g_operation(L[i, j + 1], L[i - half, j + 1], top_bit)


def _update_bits_pdf(B, x, n):
    """Algorithm 5"""
    active = [x]
    for j in range(n):
        block = 1 << (n - j)
        half = block // 2
        nxt = []
        for i in active:
            if half <= i % block:
                if np.isnan(B[i, j]) or np.isnan(B[i - half, j]):
                    continue
                B[i - half, j + 1] = int(B[i, j]) ^ int(B[i - half, j])
                B[i, j + 1] = int(B[i, j])
                nxt.extend([i, i - half])
        active = nxt


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（PDF 算法，与 Arikan 蝶形编码器配套）"""
    llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e2, 1e2)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, n] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        x = bit_reversed(phi, n)
        _update_llrs_pdf(L, B, x, n)
        if x in frozen_set:
            B[x, 0] = 0
        else:
            B[x, 0] = 0 if L[x, 0] >= 0 else 1
        _update_bits_pdf(B, x, n)

    return np.nan_to_num(B[:, 0], nan=0).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)

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
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        llr_right = g_operation(
            llr_node[:half], llr_node[half:], u_hat[bit_offset:bit_offset + half]
        )
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat
