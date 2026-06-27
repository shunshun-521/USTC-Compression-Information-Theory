"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    _compute_left_alpha,
    _compute_right_alpha,
    _compute_encoding_step,
    _position_bits,
    path_metric_update,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径分裂时深拷贝状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "intermediate_llr": [llr_ch.copy()],
            "intermediate_bits": [np.zeros(self.N, dtype=np.int8) for _ in range(self.n + 1)],
            "previous_state": np.ones(self.n, dtype=np.int8),
        }
        length = self.N // 2
        while length > 0:
            path["intermediate_llr"].append(np.zeros(length, dtype=np.float64))
            length //= 2
        return path

    def _llr_at_position(self, path, position):
        current_state = _position_bits(position, self.n)
        for i in range(1, self.n + 1):
            if current_state[i - 1] == path["previous_state"][i - 1]:
                continue
            llr = path["intermediate_llr"][i - 1]
            if current_state[i - 1] == 0:
                path["intermediate_llr"][i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (self.n - i))
                left_bits = path["intermediate_bits"][i][start:end]
                path["intermediate_llr"][i] = _compute_right_alpha(llr, left_bits)
        return float(path["intermediate_llr"][-1][0])

    def _apply_decision(self, path, position, decision):
        path["intermediate_bits"][-1][position] = decision
        for i in range(self.n - 1, -1, -1):
            path["intermediate_bits"][i] = _compute_encoding_step(
                i, self.n, path["intermediate_bits"][i + 1], path["intermediate_bits"][i]
            )
        path["previous_state"] = _position_bits(position, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for position in range(self.N):
            candidates = []
            for path in paths:
                llr = self._llr_at_position(path, position)
                if self.frozen_bits[position]:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] = path_metric_update(path["pm"], llr, 0)
                    self._apply_decision(new_path, position, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path["pm"] = path_metric_update(path["pm"], llr, bit)
                        self._apply_decision(new_path, position, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["intermediate_bits"][-1], self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        u_hat = best["intermediate_bits"][-1].astype(np.int8)
        return u_hat, best["pm"]
