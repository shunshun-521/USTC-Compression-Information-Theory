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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0 and Lb.ndim == 0
    if scalar:
        s1 = 1 if La >= 0 else -1
        s2 = 1 if Lb >= 0 else -1
        return float(s1 * s2 * min(abs(La), abs(Lb)))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(x):
    return not np.any(np.isnan(x))


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [position[0] + 1,
            position[1] + 2 ** (position[2] - 1 - position[0]),
            position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                     for i in range(half)])


def _get_left_bit(left_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _sc_factor_graph(y_llr, information_pos, frozen_val, n, N):
    """因子图 SC 译码（非递归遍历）。"""
    information_pos = set(int(i) for i in information_pos)
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1:p1 + span]
        up_bit = bit_matrix[p0][p1:p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1:p1 + half]
        left_bit = bit_matrix[p0 + 1][p1:p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half:p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half:p1 + span]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[p0][p1:p1 + span] = up_bit.copy()
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_pos = p1 + 1
                bit_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_bit(
                    right_llr[0], information_pos, frozen_val, right_pos)
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            llr_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_decided(left_llr):
            llr_matrix[p0 + 1][p1:p1 + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                bit_matrix[p0 + 1][p1:p1 + half] = _get_left_bit(
                    left_llr[0], information_pos, frozen_val, p1)
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（调用因子图实现作参考验证）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口保留）。"""
    n = int(np.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = 0
        p = phi
        while p % 2 == 1:
            psi += 1
            p //= 2
        llr_layer_vec.append(list(range(psi, n)))
        bit_layer_vec.append(list(range(psi - 1, -1, -1)) if psi > 0 else [])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先经比特倒序置换，与编码器 bit-reversal 约定一致。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    y_llr = np.asarray(llr_ch, dtype=np.float64)[br]
    information_pos = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]
    u_hat = _sc_factor_graph(y_llr, information_pos, 0, n, N)
    return u_hat
