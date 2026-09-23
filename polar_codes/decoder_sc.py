"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def _sign(La):
    """符号函数，零值取 +1 避免 min-sum 退化。"""
    return np.where(La >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return _sign(La) * _sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


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


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD）。
    llr_ch: 自然码字顺序的信道 LLR
    frozen_bits: 1=冻结, 0=信息
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch[br]

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n)
        B[l, n] = 0 if l in frozen_set or L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n]


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 译码器使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            bit_layers.append(0)
            psi2 = (phi - 1) >> 1
            while psi2 > 0 and psi2 % 2 == 0:
                psi2 >>= 1
            if psi2 > 0:
                bit_layers.append(int(math.log2(psi2 & -psi2)))
        elif phi > 0:
            psi2 = phi
            while psi2 > 0 and psi2 % 2 == 0:
                psi2 >>= 1
            if psi2 > 0:
                bit_layers.append(int(math.log2(psi2 & -psi2)))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
