"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    _compute_left_alpha,
    _compute_right_alpha,
    _compute_encoding_step,
    _position_bits,
)

CRC8_POLY = 0xE0
CRC16_POLY = 0xA001


def _crc_feedback(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) ^ (poly * feedback)) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8: 0x07; CRC-16: 0x8005（反馈多项式为反射形式）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_feedback(extended, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_feedback(bits, poly, crc_length) == 0


class _SCPath:
    __slots__ = (
        "n",
        "N",
        "info_mask",
        "intermediate_llr",
        "intermediate_bits",
        "current_state",
        "previous_state",
        "current_decision",
        "path_metric",
        "u_hat",
    )

    def __init__(self, n, N, info_mask, llr_ch):
        self.n = n
        self.N = N
        self.info_mask = info_mask
        self.intermediate_llr = [llr_ch.copy()]
        length = N // 2
        while length > 0:
            self.intermediate_llr.append(np.zeros(length, dtype=np.float64))
            length //= 2
        self.intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
        self.current_state = np.zeros(n, dtype=np.int8)
        self.previous_state = np.ones(n, dtype=np.int8)
        self.current_decision = 0
        self.path_metric = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    @property
    def current_llr(self):
        return self.intermediate_llr[-1][0]

    def copy(self):
        p = _SCPath(self.n, self.N, self.info_mask, self.intermediate_llr[0])
        p.intermediate_llr = [x.copy() for x in self.intermediate_llr]
        p.intermediate_bits = [x.copy() for x in self.intermediate_bits]
        p.current_state = self.current_state.copy()
        p.previous_state = self.previous_state.copy()
        p.current_decision = self.current_decision
        p.path_metric = self.path_metric
        p.u_hat = self.u_hat.copy()
        return p

    def update_alpha(self, position):
        self.current_state = _position_bits(position, self.n)
        for i in range(1, self.n + 1):
            if self.current_state[i - 1] == self.previous_state[i - 1]:
                continue
            llr = self.intermediate_llr[i - 1]
            if self.current_state[i - 1] == 0:
                self.intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (self.n - i))
                left_bits = self.intermediate_bits[i][start:end]
                self.intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

    def update_metric(self):
        llr = self.current_llr
        d = self.current_decision
        if llr >= 0:
            self.path_metric -= llr * d
        else:
            self.path_metric += llr * (1 - d)

    def propagate_bits(self, position):
        self.u_hat[position] = self.current_decision
        self.intermediate_bits[-1][position] = self.current_decision
        for i in range(self.n - 1, -1, -1):
            self.intermediate_bits[i] = _compute_encoding_step(
                i, self.n, self.intermediate_bits[i + 1], self.intermediate_bits[i]
            )
        self.previous_state = self.current_state.copy()


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_mask = ~self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.info_mask)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCPath(self.n, self.N, self.info_mask, llr_ch)]

        for position in range(self.N):
            for path in paths:
                path.update_alpha(position)

            if self.info_mask[position]:
                new_paths = []
                for path in paths:
                    p0 = path.copy()
                    p1 = path.copy()
                    p0.current_decision = 0
                    p1.current_decision = 1
                    p0.update_metric()
                    p1.update_metric()
                    new_paths.extend([p0, p1])
                new_paths.sort(key=lambda p: p.path_metric, reverse=True)
                paths = new_paths[: self.list_size]
            else:
                for path in paths:
                    path.current_decision = 0
                    path.update_metric()

            for path in paths:
                path.propagate_bits(position)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = max(valid, key=lambda p: p.path_metric) if valid else max(
                paths, key=lambda p: p.path_metric
            )
        else:
            best = max(paths, key=lambda p: p.path_metric)

        return best.u_hat.copy(), best.path_metric
