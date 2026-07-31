"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _combine_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _combine_right_llr(up_llr, left_bits):
    half = len(left_bits)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bits[i]) for i in range(half)])


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（树形深度优先遍历，与 sc_decode 算法等价）。
    """
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr

    def all_filled(arr):
        return not np.any(np.isnan(arr))

    def recurse(layer, start, span):
        if span == 1:
            idx = start
            llr_val = llr_matrix[layer][start]
            bit_matrix[layer][start] = 0 if frozen_bits[idx] or llr_val >= 0 else 1
            return

        half = span // 2
        up_llr = llr_matrix[layer][start:start + span]
        left_llr = llr_matrix[layer + 1][start:start + half]
        left_bit = bit_matrix[layer + 1][start:start + half]
        right_llr = llr_matrix[layer + 1][start + half:start + span]
        right_bit = bit_matrix[layer + 1][start + half:start + span]

        if not all_filled(left_llr):
            llr_matrix[layer + 1][start:start + half] = _combine_left_llr(up_llr)
            recurse(layer + 1, start, half)

        left_bit = bit_matrix[layer + 1][start:start + half]
        if all_filled(left_bit) and not all_filled(right_llr):
            llr_matrix[layer + 1][start + half:start + span] = _combine_right_llr(up_llr, left_bit)
            recurse(layer + 1, start + half, half)

        left_bit = bit_matrix[layer + 1][start:start + half]
        right_bit = bit_matrix[layer + 1][start + half:start + span]
        if all_filled(left_bit) and all_filled(right_bit) and np.any(np.isnan(bit_matrix[layer][start:start + span])):
            combined = np.array([(left_bit + right_bit) % 2, right_bit]).reshape(1, -1)
            bit_matrix[layer][start:start + span] = combined

    recurse(0, 0, N)
    return bit_matrix[n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    信道 LLR 位于第 0 层，沿层 0..n-1 递推至根节点判决。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        layers_bit = [l for l in range(n) if (phi >> l) & 1]
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（树形遍历，O(N log N)）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    def all_filled(arr):
        return not np.any(np.isnan(arr))

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    while not all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0]][start:start + span]
        left_llr = llr_matrix[position[0] + 1][start:start + span // 2]
        left_bit = bit_matrix[position[0] + 1][start:start + span // 2]
        right_llr = llr_matrix[position[0] + 1][start + span // 2:start + span]
        right_bit = bit_matrix[position[0] + 1][start + span // 2:start + span]

        if all_filled(bit_matrix[position[0]][start:start + span]):
            position = up(position)
        elif all_filled(right_bit):
            combined = np.array([(left_bit + right_bit) % 2, right_bit]).reshape(1, -1)
            bit_matrix[position[0]][start:start + span] = combined
        elif all_filled(right_llr):
            if position[0] == position[2] - 1:
                idx = position[1] + 1
                val = 0 if frozen_bits[idx] or right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][start + span // 2:start + span] = val
            else:
                position = rightdown(position)
        elif all_filled(left_bit):
            llr_matrix[position[0] + 1][start + span // 2:start + span] = _combine_right_llr(up_llr, left_bit)
        elif not all_filled(left_llr):
            llr_matrix[position[0] + 1][start:start + span // 2] = _combine_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                idx = position[1]
                val = 0 if frozen_bits[idx] or left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][start:start + span // 2] = val
            else:
                position = leftdown(position)

    return bit_matrix[n].astype(int)
