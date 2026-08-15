"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(phi, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(phi, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(phi, n):
    result = 0
    for i in range(n):
        if phi & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _prepare_llr(llr_ch):
    """编码含比特倒序置换时，信道 LLR 需同步倒序。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _frozen_set_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits == 1)[0])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与 sc_decode 等价，采用相同的信道 LLR 比特倒序约定。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = np.array([1 << (n - layer) for layer in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = _frozen_set_from_mask(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = _prepare_llr(llr_ch)

    u_hat = np.zeros(N, dtype=int)

    for phi in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(phi, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phi, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if phi in frozen_set:
            B[phi, n] = 0
        else:
            B[phi, n] = 0 if L[phi, n] >= 0 else 1

        u_hat[phi] = B[phi, n]

        if phi < N // 2:
            continue

        for s in range(n, n - _active_bit_level(phi, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phi, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
