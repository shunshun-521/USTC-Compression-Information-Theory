"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(arr):
    return not np.any(np.isnan(arr))


def _up_position(position):
    p0, p1, p2, p3 = position
    p1_new = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0 - 1, p1_new, p2, p3]


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
        dtype=np.float64,
    )


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array(
        [
            g_operation(up_llr[i], up_llr[i + half], left_bit[i])
            for i in range(half)
        ],
        dtype=np.float64,
    )


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _sc_tree_decode(y_llr, frozen_bits):
    """
    基于因子树深度优先遍历的 SC 译码。
    信道 LLR 置于第 0 层，判决结果在第 n 层。
    """
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_positions = set(np.where(~frozen_bits)[0])

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        left_llr = llr_matrix[
            position[0] + 1, position[1] : position[1] + span // 2
        ]
        left_bit = bit_matrix[
            position[0] + 1, position[1] : position[1] + span // 2
        ]
        right_llr = llr_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]
        right_bit = bit_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]

        if _all_decided(up_bit):
            position = _up_position(position)
        elif _all_decided(right_bit):
            merged = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = merged
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_positions:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_bit_val = 0
                bit_matrix[
                    position[0] + 1,
                    position[1] + span // 2 : position[1] + span,
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            new_right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[
                position[0] + 1,
                position[1] + span // 2 : position[1] + span,
            ] = new_right_llr
        elif not _all_decided(left_llr):
            new_left_llr = _get_left_llr(up_llr)
            llr_matrix[
                position[0] + 1, position[1] : position[1] + span // 2
            ] = new_left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_positions:
                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit_val = 0
                bit_matrix[
                    position[0] + 1, position[1] : position[1] + span // 2
                ] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码端含比特倒序置换，因此先将信道 LLR 做相同置换。
    冻结位索引保持自然序（与 GA 构造一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_tree_decode(llr_ch[br], frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用同一树遍历实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        p = phi
        psi = 0
        while p % 2 == 1:
            psi += 1
            p = (p - 1) // 2
        start_layer = int(math.log2(psi)) if psi > 0 else -1
        llr_layer_vec.append(list(range(n - 1, start_layer, -1)))

        if phi % 2 == 0:
            bit_layer_vec.append([])
        else:
            layers = []
            pp = phi
            while pp % 2 == 1:
                layers.append(int(math.log2(pp & -pp)))
                pp -= 1
            bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
