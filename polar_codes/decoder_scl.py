"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _sc_tree_decode, _info_positions


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    """多项式长除法求 CRC 余数（MSB first）。"""
    bits = list(int(b) for b in info_bits)
    reg = bits + [0] * crc_length
    for i in range(len(bits)):
        if reg[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    reg[i + j] ^= 1
    return np.array(reg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


def _pm_update(llr_val, bit_val):
    """路径度量：与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr_val >= 0 else 1
    if bit_val == hard:
        return 0.0
    return abs(llr_val)


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 树遍历直到完成 split_pos 处判决。"""
    N = int(bit_matrix[0].size)
    n = int(math.log2(N))
    information_pos = set(information_pos)

    def all_filled(x):
        return not np.isnan(x).any()

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    from decoder_sc import f_operation, g_operation

    def get_up_bit(left_bit, right_bit):
        length = left_bit.size
        temp = np.array([(left_bit + right_bit) % 2, right_bit])
        temp.resize((1, 2 * length))
        return temp

    def get_left_llr(up_llr):
        length = int(up_llr.size / 2)
        return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])

    def get_right_llr(left_bit, up_llr):
        length = int(left_bit.size)
        return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])

    def get_left_bit(left_llr, left_bit_pos):
        if left_bit_pos in information_pos:
            return 0 if left_llr >= 0 else 1
        return frozen_bit

    def get_right_bit(right_llr, right_bit_pos):
        if right_bit_pos in information_pos:
            return 0 if right_llr >= 0 else 1
        return frozen_bit

    # 定位起始 position
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect % 2 == 0:
        position = [n - 1, detect]
    else:
        position = [n - 1, detect - 1]
    if detect == -1:
        position = [0, 0]
    position = [position[0], position[1], n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        sl = slice(position[1], position[1] + span)
        half = span // 2
        sl_l = slice(position[1], position[1] + half)
        sl_r = slice(position[1] + half, position[1] + span)

        up_llr = llr_matrix[position[0]][sl]
        up_bit = bit_matrix[position[0]][sl]
        left_llr = llr_matrix[position[0] + 1][sl_l]
        left_bit = bit_matrix[position[0] + 1][sl_l]
        right_llr = llr_matrix[position[0] + 1][sl_r]
        right_bit = bit_matrix[position[0] + 1][sl_r]

        if all_filled(up_bit):
            position = up(position)
            continue
        if all_filled(right_bit):
            bit_matrix[position[0]][sl] = get_up_bit(left_bit, right_bit).copy()
            continue
        if all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_matrix[position[0] + 1][sl_r] = get_right_bit(right_llr, position[1] + half)
            else:
                position = rightdown(position)
            continue
        if all_filled(left_bit):
            llr_matrix[position[0] + 1][sl_r] = get_right_llr(left_bit, up_llr)
            continue
        if not all_filled(left_llr):
            llr_matrix[position[0] + 1][sl_l] = get_left_llr(up_llr)
            continue
        if position[0] == position[2] - 1:
            bit_matrix[position[0] + 1][sl_l] = get_left_bit(left_llr, position[1])
        else:
            position = leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（列表路径维护）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_pos = _info_positions(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        info_sorted = sorted(self.info_pos.tolist())

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_loc = 0
        while split_loc < len(info_sorted):
            split_pos = info_sorted[split_loc]
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to_split(
                    llr_m.copy(), bit_m.copy(), self.info_pos, 0, split_pos
                )
                llr_leaf = llr_m[n][split_pos]
                for bit in (0, 1):
                    lm = llr_m.copy()
                    bm = bit_m.copy()
                    bm[n][split_pos] = bit
                    new_pm = pm + _pm_update(llr_leaf, bit)
                    new_llr_list.append(lm)
                    new_bit_list.append(bm)
                    new_pm_list.append(new_pm)

            order = np.argsort(new_pm_list)
            llr_list = [new_llr_list[i] for i in order[: self.list_size]]
            bit_list = [new_bit_list[i] for i in order[: self.list_size]]
            pm_list = [new_pm_list[i] for i in order[: self.list_size]]
            split_loc += 1

        # 完成剩余冻结位判决
        best_idx = 0
        if self.crc_length > 0:
            for i, bm in enumerate(bit_list):
                u = bm[n].astype(int)
                info_bits = u[self.info_pos]
                if crc_check(info_bits, self.crc_length):
                    best_idx = i
                    break
        else:
            best_idx = int(np.argmin(pm_list))

        u_hat = bit_list[best_idx][n].astype(int)
        # 冻结位强制为 0
        u_hat[self.frozen_bits != 0] = 0
        return u_hat, pm_list[best_idx]
