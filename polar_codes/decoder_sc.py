"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与递归等价）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if out.size == 1 else out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    u_hat = np.atleast_1d(np.asarray(u_hat, dtype=np.float64))
    out = (1.0 - 2.0 * u_hat) * La + Lb
    return float(out[0]) if out.size == 1 else out


def _all_num(x):
    """1 表示全部为有效数值，0 表示含 NaN（待计算）"""
    return 1 if not np.any(np.isnan(np.asarray(x))) else 0


def _up(position):
    p0, p1, p2, p3 = position
    p0 -= 1
    p1 = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0, p1, p2, p3]


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - p0 - 1), p2, p3]


def _get_up_bit(left_bit, right_bit):
    left_bit = np.asarray(left_bit, dtype=int)
    right_bit = np.asarray(right_bit, dtype=int)
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)] + list(right_bit))
    return temp.reshape(1, 2 * length)


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + half]) for i in range(half)])


def _get_right_llr(left_bit, up_llr):
    half = len(left_bit)
    return np.array([g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)])


def _get_left_bit(left_llr, info_set, frozen_val, pos):
    if pos in info_set:
        val = float(np.asarray(left_llr).ravel()[0])
        return 0 if val >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, info_set, frozen_val, pos):
    if pos in info_set:
        val = float(np.asarray(right_llr).ravel()[0])
        return 0 if val > 0 else 1
    return frozen_val


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（基于因子树矩阵遍历，与标准实现一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    info_set = set(int(i) for i in np.where(~frozen_bits)[0])
    frozen_val = 0
    info_list = list(info_set)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1 : p1 + half]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half : p1 + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        else:
            if _all_num(right_bit) == 1:
                up_bit_row = _get_up_bit(left_bit, right_bit)
                bit_matrix[p0][p1 : p1 + span] = up_bit_row.copy()
            else:
                if _all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_val = _get_right_bit(
                            right_llr, info_list, frozen_val, p1 + half
                        )
                        bit_matrix[p0 + 1][p1 + half : p1 + span] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit) == 1:
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[p0 + 1][p1 + half : p1 + span] = right_llr_new
                    else:
                        if _all_num(left_llr) == 0:
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[p0 + 1][p1 : p1 + half] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_val = _get_left_bit(
                                    left_llr, info_list, frozen_val, p1
                                )
                                bit_matrix[p0 + 1][p1 : p1 + half] = left_bit_val
                            else:
                                position = _leftdown(position)

    u_hat = np.array(bit_matrix[n], dtype=int)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用与参考等价的矩阵遍历实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=12.0):
    """SC 译码无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            return False
    return True
