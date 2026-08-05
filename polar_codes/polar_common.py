"""极化码公共工具函数（f/g 运算、生成矩阵、SC 辅助函数）。"""
import numpy as np

_GENERATOR_CACHE = {}


def generate_matrix(n):
    """生成 Arikan 极化码生成矩阵 G_N = F^{\\otimes n}。"""
    if n in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[n]
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    _GENERATOR_CACHE[n] = G
    return G


def all_num(x):
    length = x.size
    for i in range(length):
        if np.isnan(x[i]):
            return 0
    return 1


def leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]


def up(position):
    p0, p1, p2, p3 = position
    p1_t = np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1))
    return [p0 - 1, int(p1_t), p2, p3]


def get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def f_hf(L1, L2):
    """min-sum 近似的 f 运算。"""
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return s1 * s2 * np.min([np.abs(L1), np.abs(L2)])


def f_hf_sms(L1, L2, alpha=0.9375):
    s1 = np.sign(L1)
    s2 = np.sign(L2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return alpha * s1 * s2 * np.min([np.abs(L1), np.abs(L2)])


def g(L1, L2, U1):
    return (1 - 2 * U1) * L1 + L2


def element_update_left(left, right, alpha=0.9375):
    value = np.zeros(2)
    value[0] = f_hf_sms(right[1] + left[1], left[0], alpha)
    value[1] = f_hf_sms(left[0], right[0], alpha) + left[1]
    return value


def element_update_right(left, right, alpha=0.9375):
    value = np.zeros(2)
    value[0] = f_hf_sms(right[1] + left[1], right[0], alpha)
    value[1] = f_hf_sms(left[0], right[0], alpha) + right[1]
    return value


def bp_update_left(left_array, right_array, left_array_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [
                    right_array[2 * i * interval + j],
                    right_array[2 * i * interval + j + interval],
                ]
            )
            get_value = element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def bp_update_right(left_array, right_array, left_array_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [
                    right_array[2 * i * interval + j],
                    right_array[2 * i * interval + j + interval],
                ]
            )
            get_value = element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def get_pm_update(llr_array, bit_array, pm_method="hf"):
    pm = 0.0
    if pm_method == "hf":
        for i in range(llr_array.size):
            hard = 0 if llr_array[i] >= 0 else 1
            if hard != bit_array[i]:
                pm += np.abs(llr_array[i])
    else:
        for i in range(llr_array.size):
            pm += np.log(1 + np.exp(-1 * (1 - 2 * bit_array[i]) * llr_array[i]))
    return pm
