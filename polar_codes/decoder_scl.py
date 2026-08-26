"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_decided,
    _combine_bits,
    _frozen_bits_to_info,
    _left_down,
    _right_down,
    _up,
    f_operation,
    g_operation,
    sc_decode_with_llr_reversal,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _get_up_loc(bit_matrix):
    n, N = bit_matrix.shape[0] - 1, bit_matrix.shape[1]
    decided = bit_matrix[n]
    for i in range(N):
        if np.isnan(decided[i]):
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect < 0:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_phi(llr_matrix, bit_matrix, information_pos, frozen_value, target_phi):
    """运行 SC 到第 target_phi 个比特判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, target_phi]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1]:position[1] + span]
        up_bit = bit_matrix[position[0], position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue

        if _all_decided(right_bit):
            up_bit = _combine_bits(left_bit, right_bit)
            bit_matrix[position[0], position[1]:position[1] + span] = up_bit
            continue

        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in information_pos:
                    right_bit_val = 0.0 if right_llr[0] > 0 else 1.0
                else:
                    right_bit_val = float(frozen_value)
                bit_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _right_down(position)
            continue

        if _all_decided(left_bit):
            right_llr = np.array([
                g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                for i in range(half)
            ])
            llr_matrix[position[0] + 1, position[1] + half:position[1] + span] = right_llr
            continue

        if not _all_decided(left_llr):
            left_llr = np.array([
                f_operation(up_llr[i], up_llr[i + half])
                for i in range(half)
            ])
            llr_matrix[position[0] + 1, position[1]:position[1] + half] = left_llr
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if left_bit_pos in information_pos:
                left_bit_val = 0.0 if left_llr[0] >= 0 else 1.0
            else:
                left_bit_val = float(frozen_value)
            bit_matrix[position[0] + 1, position[1]:position[1] + half] = left_bit_val
        else:
            position = _left_down(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos, self.frozen_value = _frozen_bits_to_info(frozen_bits)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        if self.list_size == 1:
            u_hat = sc_decode_with_llr_reversal(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        N, n = self.N, self.n
        paths = [{
            "pm": 0.0,
            "llr": np.full((n + 1, N), np.nan, dtype=np.float64),
            "bit": np.full((n + 1, N), np.nan, dtype=np.float64),
        }]
        paths[0]["llr"][0] = llr_ch

        split_positions = list(self.information_pos)

        for phi in range(N):
            if phi not in split_positions and phi != N - 1:
                for path in paths:
                    path["llr"], path["bit"] = _sc_step_to_phi(
                        path["llr"].copy(),
                        path["bit"].copy(),
                        self.information_pos,
                        self.frozen_value,
                        phi,
                    )
                continue

            candidates = []
            for path in paths:
                llr_m, bit_m = _sc_step_to_phi(
                    path["llr"].copy(),
                    path["bit"].copy(),
                    self.information_pos,
                    self.frozen_value,
                    phi,
                )
                llr_val = llr_m[0, phi]
                decided = int(bit_m[n, phi])

                if self.frozen_bits[phi]:
                    candidates.append({
                        "pm": path["pm"] + _path_metric_penalty(llr_val, 0),
                        "llr": llr_m,
                        "bit": bit_m,
                    })
                else:
                    for bit in (0, 1):
                        bit_new = bit_m.copy()
                        if bit != decided:
                            bit_new[n, phi] = float(bit)
                        candidates.append({
                            "pm": path["pm"] + _path_metric_penalty(llr_val, bit),
                            "llr": llr_m.copy(),
                            "bit": bit_new,
                        })

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p["bit"][n][self.information_pos].astype(int)
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["bit"][n].astype(int), best["pm"]
