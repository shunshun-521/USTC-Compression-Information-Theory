"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    """比特倒序索引。"""
    result = 0
    for _ in range(n):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def _active_llr_level(i, n):
    """从最高位起找第一个 1 的位置（层数）。"""
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
    """从最高位起找第一个 0 的位置（层数）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_rec(llr, frozen_bits):
    """递归 SC 译码核心（返回 u_hat 与中间阶段重编码比特）。"""
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            u = 0
        else:
            u = 0 if llr[0] >= 0 else 1
        u_arr = np.array([u], dtype=np.int8)
        return u_arr, u_arr.copy()

    half = N // 2
    llr1 = llr[:half]
    llr2 = llr[half:]

    llr_left = f_operation(llr1, llr2)
    u_left, u_left_up = _sc_decode_rec(llr_left, frozen_bits[:half])

    llr_right = g_operation(llr1, llr2, u_left_up)
    u_right, u_right_up = _sc_decode_rec(llr_right, frozen_bits[half:])

    u_hat = np.concatenate([u_left, u_right])
    u_hat1_up = (u_left_up ^ u_right_up).astype(np.int8)
    u_hat_up = np.concatenate([u_hat1_up, u_right_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
    u_hat, _ = _sc_decode_rec(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        start_bit = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, start_bit, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（比特倒序译码顺序）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits == 1)[0])
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        B[j, s] + B[j - branch_size, s]
                    ) % 2
                    B[j, s - 1] = B[j, s]

    return u_hat
