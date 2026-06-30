"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_factor_graph,
    _all_decided,
    _up,
    _leftdown,
    _rightdown,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
    _get_left_bit,
    _get_right_bit,
    f_operation,
)


def _crc_mod_bits(bits, poly, reg_bits):
    """按位 CRC 模2除法。"""
    reg = list(bits[:reg_bits])
    for b in bits[reg_bits:]:
        msb = reg.pop(0)
        reg.append(int(b) ^ msb)
        if msb:
            feedback = [(poly >> i) & 1 for i in range(reg_bits - 1, -1, -1)]
            reg = [(reg[i] ^ feedback[i]) for i in range(reg_bits)]
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_mod_bits(msg, poly, crc_length)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _get_up_loc(bit_matrix, n, N):
    detect = -1
    for i in range(N):
        if not (bit_matrix[n, i] == 0 or bit_matrix[n, i] == 1):
            detect = i - 1
            break
    if detect == -1:
        return 0, 0
    if detect % 2 == 0:
        return n - 1, detect
    return n - 1, detect - 1


def _sc_step_to_bit(y_llr, llr_matrix, bit_matrix, information_pos, frozen_val, n, N, target_bit):
    """SC 译码至 target_bit（含）。"""
    information_pos = set(int(i) for i in information_pos)
    loc_row, loc_col = _get_up_loc(bit_matrix, n, N)
    position = [loc_row, loc_col, n, N]

    while not (bit_matrix[n, target_bit] == 0 or bit_matrix[n, target_bit] == 1):
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
                bit_matrix[p0 + 1][p1 + half:p1 + span] = _get_right_bit(
                    right_llr[0], information_pos, frozen_val, p1 + 1)
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

    return llr_matrix, bit_matrix


def _path_metric_update(pm, llr, bit):
    u_hard = 0 if llr >= 0 else 1
    if bit != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        y_llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
        info_pos = list(self.information_pos)
        frozen_val = 0

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = y_llr

        paths = [{'llr': llr_matrix.copy(), 'bit': bit_matrix.copy(), 'pm': 0.0}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_m, bit_m = _sc_step_to_bit(
                    y_llr, path['llr'], path['bit'], info_pos, frozen_val, n, N, phi)
                llr_at_bit = llr_m[n, phi]
                decided = int(bit_m[n, phi])

                if self.frozen_bits[phi]:
                    pm = _path_metric_update(path['pm'], llr_at_bit, 0)
                    if decided != 0:
                        bit_m = bit_m.copy()
                        bit_m[n, phi] = 0
                    new_paths.append({'llr': llr_m, 'bit': bit_m, 'pm': pm})
                else:
                    for bit in (0, 1):
                        bm = bit_m.copy()
                        bm[n, phi] = bit
                        pm = _path_metric_update(path['pm'], llr_at_bit, bit)
                        new_paths.append({'llr': llr_m.copy(), 'bit': bm, 'pm': pm})

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = p['bit'][n].astype(int)
                info_bits = u[self.information_pos]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['bit'][n].astype(int), best['pm']
