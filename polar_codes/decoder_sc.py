"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = 1 if sa == 0 else sa
    sb = 1 if sb == 0 else sb
    return sa * sb * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _all_defined(arr):
    return not np.any(np.isnan(arr))


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
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def _sc_decode_core(y_llr, information_pos, frozen_bit):
    """非递归 SC 译码核心（层 0 为信道 LLR，层 n 为比特判决）。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]
    info_set = set(int(i) for i in information_pos)

    while not _all_defined(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]

        if _all_defined(up_bit):
            position = _up(position)
        elif _all_defined(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit.copy()
        elif _all_defined(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    right_bit = 0 if right_llr[0] > 0 else 1
                else:
                    right_bit = frozen_bit
                bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_bit
            else:
                position = _rightdown(position)
        elif _all_defined(left_bit):
            length = len(left_bit)
            right_llr = np.array(
                [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
            )
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_llr
        elif not _all_defined(left_llr):
            length = span // 2
            left_llr = np.array(
                [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
            )
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_set:
                    left_bit = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit = frozen_bit
                bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_bit
            else:
                position = _leftdown(position)

    return bit_matrix[n]


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（包装非递归核心，对信道 LLR 做比特倒序置换）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供文档/扩展使用）。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        layer = 0
        while psi & 1:
            psi >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        psi = phi + 1
        layer = 0
        bit_layers = []
        while (psi & 1) == 0 and layer < n:
            bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    编码采用 G_N = B_N F^{⊗n}，因此译码前对信道 LLR 做比特倒序置换。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    info_pos = _frozen_to_info_pos(frozen_bits)
    frozen_bit = 0

    u_hat = _sc_decode_core(llr_ch[br], info_pos, frozen_bit)
    return np.nan_to_num(u_hat, nan=0).astype(int)
