"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _all_set, _info_indices_from_frozen


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码至 split_pos 位判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(int(i) for i in information_pos)

    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        half = span // 2
        row, col = position[0], position[1]

        up_llr = llr_matrix[row, col:col + span]
        up_bit = bit_matrix[row, col:col + span]
        left_llr = llr_matrix[row + 1, col:col + half]
        left_bit = bit_matrix[row + 1, col:col + half]
        right_llr = llr_matrix[row + 1, col + half:col + span]
        right_bit = bit_matrix[row + 1, col + half:col + span]

        if _all_set(up_bit):
            new_row = row - 1
            new_col = int(
                np.floor(col / (2 ** (position[2] - row + 1)))
                * (2 ** (position[2] - row + 1))
            )
            position = [new_row, new_col, position[2], position[3]]
        elif _all_set(right_bit):
            temp = np.array([(left_bit + right_bit) % 2, right_bit])
            temp.resize((1, 2 * len(left_bit)))
            bit_matrix[row, col:col + span] = temp[0]
        elif _all_set(right_llr):
            if row == position[2] - 1:
                bit_pos = col + 1
                if bit_pos in info_set:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[row + 1, col + half:col + span] = bit_val
            else:
                position = [row + 1, col + 2 ** (position[2] - 1 - row), position[2], position[3]]
        elif _all_set(left_bit):
            right_llr_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
                dtype=np.float64,
            )
            llr_matrix[row + 1, col + half:col + span] = right_llr_new
        elif np.any(np.isnan(left_llr)):
            left_llr_new = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
                dtype=np.float64,
            )
            llr_matrix[row + 1, col:col + half] = left_llr_new
        else:
            if row == position[2] - 1:
                bit_pos = col
                if bit_pos in info_set:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_bit
                bit_matrix[row + 1, col:col + half] = bit_val
            else:
                position = [row + 1, col, position[2], position[3]]

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_idx = _info_indices_from_frozen(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        info_pos = list(self.info_idx)

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_loc = 0
        l_now = 1

        while split_loc < len(info_pos):
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]

                llr_out, bit_out = sc_stepping_decoder(
                    llr_temp, bit_temp, info_pos, 0, info_pos[split_loc]
                )

                prev_pos = info_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                cur_pos = info_pos[split_loc] + 1
                llr_seg = llr_out[n, prev_pos:cur_pos]
                bit_seg = bit_out[n, prev_pos:cur_pos]

                pm0 = pm_temp + _pm_update(llr_seg, bit_seg)
                new_llr_list.append(llr_out)
                new_bit_list.append(bit_out)
                new_pm_list.append(pm0)

                bit_wrong = bit_out.copy()
                bit_wrong[n, info_pos[split_loc]] = 1 - bit_wrong[n, info_pos[split_loc]]
                bit_seg_w = bit_wrong[n, prev_pos:cur_pos]
                pm1 = pm_temp + _pm_update(llr_seg, bit_seg_w)
                new_llr_list.append(llr_out.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm1)

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if info_pos[-1] != N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                llr_out, bit_out = sc_stepping_decoder(llr_temp, bit_temp, info_pos, 0, N - 1)
                prev_pos = info_pos[-1] + 1
                pm_list[i] += _pm_update(llr_out[n, prev_pos:N], bit_out[n, prev_pos:N])
                llr_list[i] = llr_out
                bit_list[i] = bit_out

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                info_bits = u_cand[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    best_u = u_cand
                    best_pm = pm_list[idx]
                    break
            if best_u is None:
                best_u = bit_list[order[0]][n].astype(int)
        else:
            best_u = bit_list[order[0]][n].astype(int)

        return best_u, best_pm
