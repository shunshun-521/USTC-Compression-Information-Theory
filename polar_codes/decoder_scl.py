"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _leftdown,
    _rightdown,
    _up,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation


def _crc_polynomial(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("Unsupported CRC length")
    poly = [0] * (crc_length + 1)
    for idx in loc:
        poly[idx] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=int)]
    poly = _crc_polynomial(crc_length)
    work = info_bits + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= poly[j]
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    info = bits[:-crc_length]
    encoded = crc_encode(info, crc_length)
    return np.array_equal(encoded, bits)


def _path_metric(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return abs(llr_val) if bit != hard else 0.0


def _init_matrices(llr_br, N):
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_br
    return llr_matrix, bit_matrix


def _step_to_phase(llr_matrix, bit_matrix, frozen_bits, phase):
    """
    将 SC 状态推进到第 phase 个比特判决完成。
    返回该比特的 LLR（在判决前）。
    """
    N = llr_matrix.shape[1]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    position = [0, 0, n, N]
    phase_llr = None

    while not _all_filled(bit_matrix[n, phase : phase + 1]):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        end = start + span
        half = span // 2

        up_llr = llr_matrix[position[0], start:end]
        up_bit = bit_matrix[position[0], start:end]
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : end]
        right_bit = bit_matrix[position[0] + 1, start + half : end]

        if _all_filled(up_bit):
            position = _up(position)
            continue

        if _all_filled(right_bit):
            combined = np.vstack([(left_bit + right_bit) % 2, right_bit]).reshape(-1)
            bit_matrix[position[0], start:end] = combined
            continue

        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + half
                if bit_pos == phase:
                    phase_llr = right_llr[0]
                bit = 0 if right_llr[0] > 0 else 1
                if frozen_bits[bit_pos]:
                    bit = 0
                bit_matrix[position[0] + 1, start + half : end] = bit
            else:
                position = _rightdown(position)
            continue

        if _all_filled(left_bit):
            right_vals = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ]
            )
            llr_matrix[position[0] + 1, start + half : end] = right_vals
            continue

        if not _all_filled(left_llr):
            left_vals = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1, start : start + half] = left_vals
            continue

        if position[0] == position[2] - 1:
            bit_pos = start
            if bit_pos == phase:
                phase_llr = left_llr[0]
            bit = 0 if left_llr[0] >= 0 else 1
            if frozen_bits[bit_pos]:
                bit = 0
            bit_matrix[position[0] + 1, start : start + half] = bit
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix, phase_llr


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]
        llr0, bit0 = _init_matrices(llr_br, self.N)
        paths = [{"llr": llr0, "bit": bit0, "pm": 0.0}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_m = copy.deepcopy(path["llr"])
                bit_m = copy.deepcopy(path["bit"])
                llr_m, bit_m, phase_llr = _step_to_phase(
                    llr_m, bit_m, self.frozen_bits, phi
                )
                if phase_llr is None:
                    phase_llr = 0.0

                if self.frozen_bits[phi]:
                    candidates.append(
                        {
                            "llr": llr_m,
                            "bit": bit_m,
                            "pm": path["pm"] + _path_metric(phase_llr, 0),
                        }
                    )
                else:
                    for bit in (0, 1):
                        llr_copy = copy.deepcopy(llr_m)
                        bit_copy = copy.deepcopy(bit_m)
                        bit_copy[self.n, phi] = bit
                        candidates.append(
                            {
                                "llr": llr_copy,
                                "bit": bit_copy,
                                "pm": path["pm"] + _path_metric(phase_llr, bit),
                            }
                        )

            candidates.sort(key=lambda item: item["pm"])
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda item: item["pm"])
        u_hat = best["bit"][self.n].astype(int)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                bits = path["bit"][self.n].astype(int)
                if crc_check(bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda item: item["pm"])
                u_hat = best["bit"][self.n].astype(int)

        return u_hat, best["pm"]
