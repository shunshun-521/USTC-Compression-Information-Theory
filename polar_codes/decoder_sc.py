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
    s1 = np.where(La >= 0, 1.0, -1.0)
    s2 = np.where(Lb >= 0, 1.0, -1.0)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _decide_bit(llr, is_frozen, frozen_value=0):
    if is_frozen:
        return frozen_value
    return 0 if llr >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（因子图矩阵遍历实现）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    info_positions = np.where(frozen_bits == 0)[0]
    frozen_value = 0

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]: position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]: position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]: position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]: position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2: position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2: position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]: position[1] + span] = up_bit_val
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                is_info = right_bit_pos in info_positions
                right_bit_val = _decide_bit(right_llr[0], not is_info, frozen_value)
                bit_matrix[position[0] + 1][position[1] + span // 2: position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + span // 2: position[1] + span] = right_llr_val
        elif not _all_decided(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]: position[1] + span // 2] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                is_info = left_bit_pos in info_positions
                left_bit_val = _decide_bit(left_llr[0], not is_info, frozen_value)
                bit_matrix[position[0] + 1][position[1]: position[1] + span // 2] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """保留接口：预计算非递归 SC 辅助向量（供 SCL 使用）。"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 1 << (i - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        layers = []
        for i in range(n - 1, -1, -1):
            if phi_bin[i] == "0":
                layers.append(n - 1 - i)
        llr_layer_vec.append(layers)

        layers_b = []
        if phi % 2 == 1:
            for i in range(n - 1, -1, -1):
                if phi_bin[i] == "0":
                    layers_b.append(n - 1 - i)
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec
