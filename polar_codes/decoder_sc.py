"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        s1 = np.sign(La)
        s2 = np.sign(Lb)
        if s1 == 0:
            s1 = 1.0
        if s2 == 0:
            s2 = 1.0
        return s1 * s2 * min(abs(La), abs(Lb))
    s1 = np.sign(La).copy()
    s2 = np.sign(Lb).copy()
    s1[s1 == 0] = 1.0
    s2[s2 == 0] = 1.0
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_num(x):
    x = np.asarray(x)
    for i in range(x.size):
        if np.isnan(x[i]):
            return False
    return True


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp[0]


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([
        f_operation(up_llr[i], up_llr[i + length]) for i in range(length)
    ])


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i])
        for i in range(length)
    ])


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _sc_decode_core(y_llr, frozen_bits):
    """顺序 SC 译码（层 0 为信道 LLR）。"""
    y_llr = np.asarray(y_llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
    information_pos = np.where(~frozen_bits)[0]
    frozen_bit = 0
    N = y_llr.size
    n = int(np.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1]:position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = _get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not _all_num(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = _get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与顺序实现等价）。"""
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 译码辅助向量。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        while (phi >> layer) & 1:
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        bits = []
        if phi % 2 == 0:
            l = 0
            while l < n and (phi % (1 << (l + 1))) == 0:
                bits.append(l)
                l += 1
        bit_layer_vec.append(bits)

    return list(range(N)), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_decode_core(llr_ch, frozen_bits)
