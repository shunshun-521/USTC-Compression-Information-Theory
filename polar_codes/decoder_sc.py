"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _llr_to_decoder(llr_ch):
    """将自然顺序信道 LLR 转为 SC/SCL 译码器所需顺序。"""
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[brp]


def bit_reversal_permutation(N):
    n = int(math.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def _h_decision(llr_val):
    return 1 if llr_val < 0 else 0


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考 aff3ct naive 结构）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(lam, offset):
        n = len(lam)
        if n == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = _h_decision(lam[0])
            return
        half = n // 2
        lam_left = f_operation(lam[:half], lam[half:])
        decode_node(lam_left, offset)
        u_left = u_hat[offset : offset + half].copy()
        lam_right = g_operation(lam[:half], lam[half:], u_left)
        decode_node(lam_right, offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layers = list(range(l, n))
        llr_layer_vec.append(llr_layers)
        l = 0
        while l < n and ((phi >> l) & 1):
            bit_layers.append(l)
            l += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前委托给递归实现，接口保持一致）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（路径更新版，供对照）。"""
    llr_ch = _llr_to_decoder(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    assert N == (1 << n)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        for li in range(l, n):
            stride = 1 << (n - 1 - li)
            if (phi % (2 * stride)) < stride:
                P[li, phi] = f_operation(
                    P[li + 1, phi], P[li + 1, phi + stride]
                )
        for li in range(l):
            stride = 1 << li
            P[li, phi] = g_operation(
                P[li + 1, phi - stride],
                P[li + 1, phi],
                C[li, phi - stride],
            )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        l = 0
        while l < n and ((phi >> l) & 1):
            C[l + 1, phi] = C[l, phi] ^ C[l, phi - (1 << l)]
            l += 1

    return u_hat
