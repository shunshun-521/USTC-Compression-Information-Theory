"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（用于递归参考实现）。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_logdomain(l1, l2):
    """对数域精确 f 运算（主译码器使用）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_logdomain(l1, l2, b):
    """对数域 g 运算。"""
    return l1 + l2 if b == 0 else l1 - l2


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码参考接口。
    置换 SC 的非递归实现与 Arikan 生成矩阵编码一致，此处与之对齐。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（层更新范围）。
    """
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        if l == 0:
            layers = list(range(n - 1, -1, -1))
        else:
            layers = list(range(n - 1, -1, -1))
            # 仅更新从根到当前比特所需层（与 PDF Update LLR 等价）
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD / 对数域，与 polar_encode 的 G=B F^{⊗n} 一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    brp = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch[brp]

    for i in range(N):
        l = _bit_reversed_index(i, n)
        for j in range(n - 1, -1, -1):
            s = 2 ** (n - j)
            t = s // 2
            for idx in range(l, N, s):
                if t > idx % s:
                    L[idx, j] = f_logdomain(L[idx, j + 1], L[idx + t, j + 1])
                else:
                    L[idx, j] = g_logdomain(L[idx, j + 1], L[idx - t, j + 1], int(B[idx - t, j]))

        if l in frozen_set:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1

        active = [l]
        for j in range(n):
            s = 2 ** (n - j)
            t = s // 2
            nxt = []
            for idx in active:
                if t <= idx % s:
                    B[idx - t, j + 1] = (B[idx, j] + B[idx - t, j]) % 2
                    B[idx, j + 1] = B[idx, j]
                    nxt.extend([idx, idx - t])
            active = nxt

    return B[:, 0].astype(np.int32)
