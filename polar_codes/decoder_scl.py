"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _all_computed,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    remainder = 0
    for bit in np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]):
        remainder ^= int(bit) << (crc_length - 1)
        if remainder & (1 << (crc_length - 1)):
            remainder = ((remainder << 1) ^ poly) & mask
        else:
            remainder = (remainder << 1) & mask
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    preferred = 0 if llr >= 0 else 1
    return 0.0 if bit == preferred else abs(llr)


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect < 0:
        return [0, 0, n]
    loc = [0, 0, n]
    for i in range(n):
        if detect % 2 == 1:
            loc[0] += 1
            loc[1] += 2 ** (n - 1 - i)
        detect //= 2
    return loc


def _sc_step_to(llr_matrix, bit_matrix, frozen_bits, stop_pos):
    """树遍历 SC，译码至 stop_pos（含）"""
    llr_matrix = llr_matrix.copy()
    bit_matrix = bit_matrix.copy()
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, stop_pos]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            bit_matrix[position[0], position[1] : position[1] + span] = _get_up_bit(
                left_bit.astype(int), right_bit.astype(int)
            )
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + half
                val = 0 if frozen_bits[pos] else (0 if right_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = _get_right_llr(
                left_bit.astype(int), up_llr
            )
        elif not _all_computed(left_llr):
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                pos = position[1]
                val = 0 if frozen_bits[pos] else (0 if left_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1, position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化：路径分裂时复制矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = llr_ch[self.br].copy()
        N, n = self.N, self.n

        llr0 = np.full((n + 1, N), np.nan)
        bit0 = np.full((n + 1, N), np.nan)
        llr0[0] = llr

        paths = [(llr0, bit0, 0.0)]

        prev_pos = -1
        for pos in self.info_positions:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_to(llr_m, bit_m, self.frozen_bits, pos)
                llr_val = llr_m[n, pos]
                bit_val = int(bit_m[n, pos])

                new_paths.append((llr_m.copy(), bit_m.copy(), pm))

                if not self.frozen_bits[pos]:
                    bit_m_alt = bit_m.copy()
                    bit_m_alt[n, pos] = 1 - bit_val
                    pm_alt = pm + _pm_penalty(llr_val, 1 - bit_val)
                    new_paths.append((llr_m.copy(), bit_m_alt, pm_alt))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]
            prev_pos = pos

        for llr_m, bit_m, pm in paths:
            llr_m, bit_m = _sc_step_to(llr_m, bit_m, self.frozen_bits, N - 1)

        if self.crc_length > 0:
            valid = []
            for llr_m, bit_m, pm in paths:
                u_hat = bit_m[n].astype(int)
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((u_hat, pm))
            if valid:
                u_hat, pm = min(valid, key=lambda x: x[1])
                return u_hat, pm

        best = min(paths, key=lambda x: x[2])
        return best[1][n].astype(int), best[2]
