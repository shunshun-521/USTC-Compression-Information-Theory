"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（P/C 数组高效实现）
"""
import math
import numpy as np


def _bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
    return result


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        sa = 1.0 if La >= 0 else -1.0
        sb = 1.0 if Lb >= 0 else -1.0
        if La == 0:
            sa = 1.0
        if Lb == 0:
            sb = 1.0
        return sa * sb * min(abs(La), abs(Lb))
    La = np.atleast_1d(La)
    Lb = np.atleast_1d(Lb)
    s1 = np.sign(La).copy()
    s2 = np.sign(Lb).copy()
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return np.ravel(s1 * s2 * np.minimum(np.abs(La), np.abs(Lb)))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0 and u_hat.ndim == 0:
        return (1.0 - 2.0 * u_hat) * La + Lb
    La = np.atleast_1d(La)
    Lb = np.atleast_1d(Lb)
    u_hat = np.atleast_1d(u_hat)
    return np.ravel((1.0 - 2.0 * u_hat) * La + Lb)


def _active_llr_level(i, n):
    """LLR 更新活跃层数"""
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
    """比特回传活跃层数"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，Arikan 树形分解）。

    注：在 x = u @ F^{⊗n} 编码约定下，生产环境请使用 sc_decode（P/C 非递归实现），
    其与编码器及 SCL/BP 译码器完全配套。递归版本展示 SC 的分治结构。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        llr_node = np.atleast_1d(np.asarray(llr_node, dtype=np.float64))
        frozen_node = np.asarray(frozen_node, dtype=bool)
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = n // 2
        llr_left = np.atleast_1d(f_operation(llr_node[:half], llr_node[half:]))
        u_left = decode_node(llr_left, frozen_node[:half])
        llr_right = np.atleast_1d(
            g_operation(llr_node[:half], llr_node[half:], u_left)
        )
        u_right = decode_node(llr_right, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


# ==================== 非递归 SC 译码（P/C 数组实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量：
      - lambda_offset[layer]: 第 layer 层的 LLR 存储偏移
      - llr_layer_vec[phi]: 第 phi 个比特需要执行 LLR 运算的层列表
      - bit_layer_vec[phi]: 第 phi 个比特需要执行比特返回的层列表
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            start = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（P/C 数组实现，按比特倒序逐位译码）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = float(f_operation(L[j, s], L[j + branch_size, s]))
                else:
                    L[j, s + 1] = float(
                        g_operation(
                            L[j - branch_size, s],
                            L[j, s],
                            B[j - branch_size, s + 1],
                        )
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            B[j, s] ^ B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n]


# ==================== 因子图矩阵遍历 SC（供 SCL 复用）====================


def _all_ready(x):
    return not np.any(np.isnan(x))


def _left_down(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _right_down(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _decide_bit(llr_val, bit_pos, info_set):
    if bit_pos in info_set:
        return 0 if llr_val >= 0 else 1
    return 0


def sc_decode_nonrecursive(llr_ch, frozen_bits, info_indices=None):
    """非递归 SC 译码（因子图矩阵遍历，供 SCL 逐步译码复用）。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    y_llr = np.asarray(llr_ch, dtype=np.float64)

    if info_indices is None:
        info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_ready(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[
            position[0] + 1, position[1] + half : position[1] + span
        ]
        right_bit = bit_matrix[
            position[0] + 1, position[1] + half : position[1] + span
        ]

        if _all_ready(up_bit):
            position = _up(position)
        elif _all_ready(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = up_bit_val.copy()
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1, position[1] + half] = _decide_bit(
                    right_llr[0], right_bit_pos, info_set
                )
            else:
                position = _right_down(position)
        elif _all_ready(left_bit):
            right_llr_val = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ]
            )
            llr_matrix[
                position[0] + 1, position[1] + half : position[1] + span
            ] = right_llr_val
        elif not _all_ready(left_llr):
            left_llr_val = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1, position[1]] = _decide_bit(
                    left_llr[0], left_bit_pos, info_set
                )
            else:
                position = _left_down(position)

    return np.nan_to_num(bit_matrix[n], nan=0).astype(np.int8)
