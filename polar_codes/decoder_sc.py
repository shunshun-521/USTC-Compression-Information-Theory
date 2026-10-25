"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """f 运算（boxplus；大 LLR 时用 min-sum 保持稳定）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    ms = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    eps = 1e-12
    ta = np.tanh(La / 2.0)
    tb = np.tanh(Lb / 2.0)
    bp = 2.0 * np.arctanh(np.clip(ta * tb, -1.0 + eps, 1.0 - eps))
    use_ms = (np.abs(La) > 30.0) | (np.abs(Lb) > 30.0)
    out = bp.copy()
    out[use_ms] = ms[use_ms]
    return out[0] if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        ll = []
        if phi == 0:
            ll = list(range(n))
        else:
            p = phi
            layer = 0
            while (p & 1) == 1 and layer < n:
                ll.append(layer)
                layer += 1
                p >>= 1
            if layer < n:
                ll.append(layer)
        llr_layer_vec.append(ll)

        bl = []
        p = phi
        layer = 0
        while (p & 1) == 0 and layer < n:
            bl.append(layer)
            layer += 1
            p >>= 1
        bit_layer_vec.append(bl)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（增量更新，供性能对比）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    m = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    P = np.zeros((m + 1, N), dtype=np.float64)
    C = np.zeros((m + 1, N), dtype=np.int8)
    P[m, :] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if phi == 0:
            span = 1
            for lam in range(m - 1, -1, -1):
                span <<= 1
                half = span // 2
                for beta in range(0, N, span):
                    for alpha in range(half):
                        i = beta + alpha
                        i2 = i + half
                        P[lam, i] = f_operation(P[lam + 1, i], P[lam + 1, i2])
                        P[lam, i2] = g_operation(P[lam + 1, i], P[lam + 1, i2], C[lam + 1, i])
        else:
            lam = 0
            p = phi
            while (p & 1) == 1 and lam < m:
                lam += 1
                p >>= 1
            lam = min(lam, m - 1)
            span = 1 << (m - lam)
            half = span // 2
            beta = (phi // span) * span
            alpha = phi - beta
            if alpha >= half:
                P[lam, phi] = g_operation(
                    P[lam + 1, phi - half], P[lam + 1, phi], C[lam + 1, phi - half]
                )

        u_hat[phi] = 0 if frozen_bits[phi] or P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        lam = 0
        p = phi
        while (p & 1) == 0 and lam < m:
            step = 1 << lam
            beta = (phi // (2 * step)) * (2 * step)
            alpha = phi % step
            i = beta + alpha
            C[lam + 1, i] = (C[lam, i] ^ C[lam, i + step]) % 2
            C[lam + 1, i + step] = C[lam, i + step]
            lam += 1
            p >>= 1

    return u_hat


sc_decode = sc_decode_recursive


sc_decode = sc_decode_recursive
