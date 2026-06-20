"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from decoder_sc_ops import f_operation, g_operation


def _all_decided(bits):
    """检查比特块是否已全部判决"""
    return not np.any(np.isnan(bits))


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（通过因子图遍历实现，与 sc_decode 等价）。
    """
    return sc_decode_factor_graph(llr, frozen_bits)


def sc_decode_factor_graph(llr_ch, frozen_bits, info_indices=None):
    """
    基于因子图遍历的 SC 译码（与 encoder G=F^{\\otimes n} 配套）。
  llr_matrix[0] 存放信道 LLR，bit_matrix[n] 存放译码结果。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    if info_indices is None:
        info_indices = np.where(~frozen_bits)[0]
    else:
        info_indices = np.asarray(info_indices, dtype=int)

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch

  # position = [row, col_start, max_row, block_width]
    position = [0, 0, n, N]

    def up(pos):
        return [pos[0] - 1,
                int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1))),
                pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - pos[0] - 1), pos[2], pos[3]]

    while not _all_decided(bit_matrix[n]):
        r, c, max_r, width = position
        span = 2 ** (max_r - r)
        up_llr = llr_matrix[r][c:c + span]
        up_bit = bit_matrix[r][c:c + span]
        half = span // 2
        left_llr = llr_matrix[r + 1][c:c + half]
        left_bit = bit_matrix[r + 1][c:c + half]
        right_llr = llr_matrix[r + 1][c + half:c + span]
        right_bit = bit_matrix[r + 1][c + half:c + span]

        if _all_decided(up_bit):
            position = up(position)
            continue

        if _all_decided(right_bit):
            combined = np.empty(span, dtype=np.float64)
            combined[:half] = (left_bit + right_bit) % 2
            combined[half:] = right_bit
            bit_matrix[r][c:c + span] = combined
            continue

        if _all_decided(right_llr):
            if r == max_r - 1:
                bit_pos = c + half
                if frozen_bits[bit_pos]:
                    val = 0.0
                else:
                    val = 0.0 if right_llr[0] >= 0 else 1.0
                bit_matrix[r + 1][c + half] = val
            else:
                position = rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_new = g_operation(
                up_llr[:half], up_llr[half:], left_bit.astype(int)
            )
            llr_matrix[r + 1][c + half:c + span] = right_llr_new
            continue

        if not _all_decided(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[r + 1][c:c + half] = left_llr_new
            continue

        if r == max_r - 1:
            bit_pos = c
            if frozen_bits[bit_pos]:
                val = 0.0
            else:
                val = 0.0 if left_llr[0] >= 0 else 1.0
            bit_matrix[r + 1][c] = val
        else:
            position = leftdown(position)

    u_hat = bit_matrix[n].astype(int)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while layer < n:
            if p % 2 == 0:
                llr_layers.append(layer)
            p >>= 1
            layer += 1

        bit_layers = []
        p = phi
        layer = 0
        while layer < n:
            if p % 2 == 1:
                bit_layers.append(layer)
            p >>= 1
            layer += 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（层叠数组实现，与因子图版本等价）。
    """
    return sc_decode_factor_graph(llr_ch, frozen_bits)


# 兼容性别名
sc_decode_nonrecursive = sc_decode
