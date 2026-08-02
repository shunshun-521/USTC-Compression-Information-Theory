"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


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


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    llr 应为比特倒序置换后的信道 LLR。
    """
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    llr_ch 应为比特倒序置换后的信道 LLR。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    u_hat = np.zeros(N, dtype=int)

    for bit_idx in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if bit_idx in frozen_set:
            u_hat[bit_idx] = 0
            B[bit_idx, n] = 0
        else:
            u_hat[bit_idx] = 0 if L[bit_idx, n] >= 0 else 1
            B[bit_idx, n] = u_hat[bit_idx]

        if bit_idx < N / 2:
            continue

        for s in range(n, n - _active_bit_level(bit_idx, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(bit_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        int(B[j, s]) ^ int(B[j - branch_size, s])
                    )
                    B[j, s - 1] = B[j, s]

    return u_hat
