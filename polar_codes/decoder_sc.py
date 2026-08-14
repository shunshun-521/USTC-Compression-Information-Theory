"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _boxplus_f(La, Lb):
    """SC 译码内部使用的精确 box-plus f 运算（log-domain）。"""
    if np.isscalar(La) and np.isscalar(Lb):
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits == 1)[0])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（min-sum f 运算）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_set = _frozen_indices(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = _boxplus_f(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i:i + 1], bit_offset + i)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Tal/Casini 索引表）。
    """
    n = int(math.log2(N))
    lambda_offset = [2 ** i - 1 for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi > 0 and (psi & 1) == 0:
            layer = int(math.log2(psi & -psi))
            layers_llr.append(layer)
            psi >>= 1

        layers_bit = []
        psi2 = phi
        while psi2 > 0 and (psi2 & 1) == 1:
            layer = int(math.log2(psi2 & -psi2))
            layers_bit.append(layer)
            psi2 >>= 1

        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive_table(llr_ch, frozen_bits):
    """
    基于预计算索引表的 SC 译码（与 sc_decode 等价接口）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    P = np.zeros(2 * N - 1, dtype=np.float64)
    C = np.zeros(2 * N - 1, dtype=int)
    u_hat = np.zeros(N, dtype=int)

    rev = bit_reversal_permutation(N)
    P[lambda_offset[n]:lambda_offset[n] + N] = llr_ch[rev]

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            offset = lambda_offset[layer]
            step = 2 ** layer
            is_even = (phi >> layer) & 1 == 0
            for beta in range(step):
                if is_even:
                    La = P[offset + beta]
                    Lb = P[offset + beta + step]
                    P[offset + beta] = _boxplus_f(La, Lb)
                else:
                    beta_offset = beta // 2
                    La = P[offset + beta_offset]
                    Lb = P[offset + beta_offset + step]
                    u_val = C[offset + beta_offset]
                    P[offset + beta] = g_operation(La, Lb, u_val)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0] >= 0 else 1

        C[0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            offset = lambda_offset[layer]
            step = 2 ** layer
            is_even = (phi >> layer) & 1 == 0
            for beta in range(step):
                if is_even:
                    C[offset + beta] ^= C[offset + beta + step]
                else:
                    beta_offset = beta // 2
                    C[offset + beta_offset + step] ^= C[offset + beta_offset]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（矩阵更新实现，与编码器比特倒序约定一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_indices(frozen_bits)

    rev = bit_reversal_permutation(N)
    llr_work = llr_ch[rev]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_work

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _boxplus_f(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
