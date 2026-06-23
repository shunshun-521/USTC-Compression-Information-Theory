"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import sc_decode, _all_filled, _leftdown, _rightdown, _up
from decoder_sc import _get_up_bit, _get_right_llr, _get_left_llr, _decide_bit


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _get_up_loc(bit_matrix, n, N):
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to(llr_matrix, bit_matrix, info_positions, split_pos):
    """运行 SC 直到比特 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]
    max_iter = N * n * 4
    iters = 0

    while np.isnan(bit_matrix[n, split_pos]) and iters < max_iter:
        iters += 1
        if position[0] < 0 or position[0] > n:
            break

        span = 2 ** (position[2] - position[0])
        sl = slice(position[1], position[1] + span)
        half = span // 2
        left_sl = slice(position[1], position[1] + half)
        right_sl = slice(position[1] + half, position[1] + span)

        up_bit = bit_matrix[position[0]][sl]

        if _all_filled(up_bit):
            if position[0] == 0:
                break
            position = _up(position)
            continue

        up_llr = llr_matrix[position[0]][sl]
        left_llr = llr_matrix[position[0] + 1][left_sl]
        left_bit = bit_matrix[position[0] + 1][left_sl]
        right_llr = llr_matrix[position[0] + 1][right_sl]
        right_bit = bit_matrix[position[0] + 1][right_sl]

        if _all_filled(right_bit):
            bit_matrix[position[0]][sl] = _get_up_bit(left_bit, right_bit).copy()
            continue
        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = position[1] + 1
                bit_matrix[position[0] + 1, position[1] + half] = _decide_bit(
                    float(right_llr[0]), pos in info_positions
                )
            else:
                position = _rightdown(position)
            continue
        if _all_filled(left_bit):
            llr_matrix[position[0] + 1][right_sl] = _get_right_llr(left_bit, up_llr)
            continue
        if not _all_filled(left_llr):
            llr_matrix[position[0] + 1][left_sl] = _get_left_llr(up_llr)
            continue
        if position[0] == position[2] - 1:
            pos = position[1]
            bit_matrix[position[0] + 1, position[1]] = _decide_bit(
                float(left_llr[0]), pos in info_positions
            )
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = list(np.where(~self.frozen_bits)[0])

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n = self.N, self.n
        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        paths = [(llr0.copy(), bit0.copy(), 0.0)]
        prev = -1

        for split_pos in self.info_positions:
            new_paths = []
        for llr_m, bit_m, pm in paths:
            llr_m, bit_m = _sc_step_to(
                llr_m.copy(), bit_m.copy(), set(self.info_positions), split_pos
            )
            llr_val = llr_m[n, split_pos]
            bit0_val = int(bit_m[n, split_pos]) if not np.isnan(bit_m[n, split_pos]) else (
                0 if llr_val >= 0 else 1
            )

            if split_pos not in self.info_positions:
                bm = bit_m.copy()
                bm[n, split_pos] = 0
                new_paths.append((llr_m.copy(), bm, pm + _pm_penalty(llr_val, 0)))
            else:
                for bit in (0, 1):
                    bm = bit_m.copy()
                    bm[n, split_pos] = bit
                    new_paths.append((llr_m.copy(), bm, pm + _pm_penalty(llr_val, bit)))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]
            prev = split_pos

        if prev < N - 1:
            finalized = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step_to(
                    llr_m.copy(), bit_m.copy(), set(self.info_positions), N - 1
                )
                finalized.append((llr_m, bit_m, pm))
            paths = finalized

        if self.crc_length > 0:
            valid = []
            for _, bit_m, pm in paths:
                u = bit_m[n].astype(np.int8)
                payload = u[self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append((u, pm))
            if valid:
                best = min(valid, key=lambda x: x[1])
                return best[0], best[1]

        best = min(paths, key=lambda x: x[2])
        return best[1][n].astype(np.int8), best[2]
