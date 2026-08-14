"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_tree_decode,
    _prepare_llr,
    _info_positions,
    f_operation,
    g_operation,
)


# CRC 多项式位位置（含最高位）
_CRC_POLY = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly_bits(crc_length):
    if crc_length not in _CRC_POLY:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    loc = _CRC_POLY[crc_length]
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    p = _crc_poly_bits(crc_length)
    crc_n = crc_length
    data = info_bits + [0] * crc_n
    q = []
    for i in range(len(info_bits)):
        if data[i] == 1:
            q.append(1)
            for j in range(crc_n + 1):
                data[j + i] ^= p[j]
        else:
            q.append(0)
    check = data[-crc_n:]
    return np.array(info_bits + check, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).tolist()
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length).tolist()
    return expected == bits


def _pm_update(llr_array, bit_array):
    """路径度量更新（hf 近似）。"""
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, frozen_value, split_pos):
    """SC 译码至 split_pos 位判决完成。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    info_set = set(information_pos.tolist())

    from decoder_sc import (
        _all_computed, _leftdown, _rightdown, _up,
        _get_up_bit, _get_left_llr, _get_right_llr, _decide_bit,
    )

    detect = -1
    for i in range(N):
        if bit_matrix[n][i] != 0 and bit_matrix[n][i] != 1:
            detect = i - 1
            break

    if detect % 2 == 0:
        loc_row, loc_col = n - 1, detect
    else:
        loc_row, loc_col = n - 1, detect - 1
    if detect == -1:
        loc_row, loc_col = 0, 0

    position = [loc_row, loc_col, n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _decide_bit(right_llr[0], right_bit_pos, info_set, frozen_value)
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
        elif not _all_computed(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _decide_bit(left_llr[0], left_bit_pos, info_set, frozen_value)
                bit_matrix[position[0] + 1][position[1]:position[1] + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = _info_positions(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n
        info_pos = self.info_positions
        L = self.list_size

        if L == 1:
            u_hat = _sc_tree_decode(y_llr, info_pos, frozen_value=0)
            return u_hat, 0.0

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_positions = info_pos.tolist()
        if split_positions[-1] != N - 1:
            split_positions = split_positions + [N - 1]

        prev = -1
        for split_pos in split_positions:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for idx in range(len(llr_list)):
                lm, bm, pm = llr_list[idx], bit_list[idx], pm_list[idx]
                lm_out, bm_out = _sc_step_to_split(
                    lm.copy(), bm.copy(), info_pos, 0, split_pos
                )
                llr_slice = lm_out[n][prev + 1:split_pos + 1]
                bit_slice = bm_out[n][prev + 1:split_pos + 1]

                new_llr_list.append(lm_out)
                new_bit_list.append(bm_out)
                new_pm_list.append(pm + _pm_update(llr_slice, bit_slice))

                bm_wrong = bm_out.copy()
                bm_wrong[n][split_pos] = 1 - bm_wrong[n][split_pos]
                wrong_slice = bm_wrong[n][prev + 1:split_pos + 1]
                new_llr_list.append(lm_out.copy())
                new_bit_list.append(bm_wrong)
                new_pm_list.append(pm + _pm_update(llr_slice, wrong_slice))

            order = np.argsort(new_pm_list)
            keep = order[:L]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            prev = split_pos

        best_idx = 0
        if self.crc_length > 0:
            sorted_idx = np.argsort(pm_list)
            for idx in sorted_idx:
                u_candidate = bit_list[idx][n].astype(int)
                u_candidate = np.array([0 if v == 0 else 1 for v in u_candidate], dtype=int)
                info_bits = u_candidate[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    best_idx = idx
                    break
            else:
                best_idx = sorted_idx[0]

        u_hat = bit_list[best_idx][n]
        u_hat = np.array([0 if v == 0 else 1 for v in u_hat], dtype=int)
        return u_hat, pm_list[best_idx]
