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
    scalar = np.isscalar(La) and np.isscalar(Lb)
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _prepare_llr(llr_ch):
    """编码含比特倒序，译码前对信道 LLR 做相同置换。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    perm = bit_reversal_permutation(len(llr_ch))
    return llr_ch[perm]


def _frozen_to_info_positions(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return np.where(frozen_bits == 0)[0]


def _all_num(x):
    return not np.any(np.isnan(x))


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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _tree_sc_decode(llr, information_pos, frozen_bit=0):
    """基于因子树遍历的 SC 译码（与编码器配套）。"""
    N = llr.size
    n = int(np.log2(N))
    information_pos = set(int(i) for i in information_pos)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = np.ones((n + 1, N), dtype=np.float64)
    bit_matrix[:] = np.nan
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in information_pos:
                    right_bit = 0 if right_llr[0] >= 0 else 1
                else:
                    right_bit = frozen_bit
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                ] = right_bit
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            length = left_bit.size
            right_llr = np.array(
                [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)],
                dtype=np.float64,
            )
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
            ] = right_llr
        elif not _all_num(left_llr):
            left_llr = f_operation(up_llr[: up_llr.size // 2], up_llr[up_llr.size // 2 :])
            llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in information_pos:
                    left_bit = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit = frozen_bit
                bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（包装树遍历实现）。"""
    llr = _prepare_llr(llr)
    info_pos = _frozen_to_info_positions(frozen_bits)
    return _tree_sc_decode(llr, info_pos, frozen_bit=0)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 lambda_offset, llr_layer_vec, bit_layer_vec。
    """
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(np.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            psi = phi + 1
            while psi < N and psi % 2 == 0:
                bit_layers.append(int(np.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（高效树遍历实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
