"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    return int(f"{x:0{n}b}"[::-1], 2)


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


def _layer_ready(arr):
    return not np.isnan(arr).any()


def _up_position(pos):
    step = 2 ** (pos[2] - pos[0] + 1)
    new_col = int(np.floor(pos[1] / step) * step)
    return [pos[0] - 1, new_col, pos[2], pos[3]]


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * len(left_bit)))
    return temp[0]


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return f_operation(up_llr[:half], up_llr[half:])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return g_operation(up_llr[:half], up_llr[half:], left_bit.astype(int))


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（因子图深度优先遍历，L[:, 0] 为信道 LLR）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    while not _layer_ready(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        sl = slice(position[1], position[1] + span)
        sl_l = slice(position[1], position[1] + span // 2)
        sl_r = slice(position[1] + span // 2, position[1] + span)

        up_llr = llr_matrix[position[0]][sl]
        up_bit = bit_matrix[position[0]][sl]
        left_llr = llr_matrix[position[0] + 1][sl_l]
        left_bit = bit_matrix[position[0] + 1][sl_l]
        right_llr = llr_matrix[position[0] + 1][sl_r]
        right_bit = bit_matrix[position[0] + 1][sl_r]

        if _layer_ready(up_bit):
            position = _up_position(position)
        elif _layer_ready(right_bit):
            bit_matrix[position[0]][sl] = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
        elif _layer_ready(right_llr):
            if position[0] == position[2] - 1:
                idx = position[1] + 1
                if idx in info_indices:
                    bit_matrix[position[0] + 1][sl_r] = 0 if right_llr[0] >= 0 else 1
                else:
                    bit_matrix[position[0] + 1][sl_r] = 0
            else:
                position = _rightdown(position)
        elif _layer_ready(left_bit):
            llr_matrix[position[0] + 1][sl_r] = _get_right_llr(left_bit, up_llr)
        elif not _layer_ready(left_llr):
            llr_matrix[position[0] + 1][sl_l] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                idx = position[1]
                if idx in info_indices:
                    bit_matrix[position[0] + 1][sl_l] = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_matrix[position[0] + 1][sl_l] = 0
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """保留接口：非递归实现使用按位倒序索引遍历。"""
    n = int(np.log2(N))
    return (
        np.array([1 << i for i in range(n + 1)], dtype=int),
        [_bit_reversed(i, n) for i in range(N)],
        None,
    )


def _update_llrs(L, B, l, n, N):
    start = n - _active_llr_level(l, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    stop = n - _active_bit_level(l, n)
    for s in range(n, stop, -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按位倒序逐比特更新 LLR 与部分和）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return u_hat


def path_metric_update(pm, llr, u):
    """路径度量更新"""
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm
