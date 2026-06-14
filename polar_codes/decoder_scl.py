"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import zlib

import numpy as np

from decoder_sc import (
    _compute_encoding_step,
    _compute_left_alpha,
    _compute_right_alpha,
    _position_state,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _digest(info_bits, crc_length):
    """基于 CRC-8/16 多项式的校验_digest"""
    data = np.packbits(np.asarray(info_bits, dtype=np.uint8).ravel())
    if crc_length == 8:
        val = 0
        poly = CRC8_POLY
        for byte in data.tobytes():
            val ^= byte
            for _ in range(8):
                if val & 0x80:
                    val = ((val << 1) ^ poly) & 0xFF
                else:
                    val = (val << 1) & 0xFF
        return val
    return zlib.crc32(data.tobytes()) & 0xFFFF


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    val = _digest(info_bits, crc_length)
    crc = np.array([(val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) <= crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "intermediate_llr", "intermediate_bits", "previous_state", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.intermediate_llr = [llr_ch.copy()]
        length = N // 2
        while length > 0:
            self.intermediate_llr.append(np.zeros(length, dtype=np.float64))
            length //= 2
        self.intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
        self.previous_state = np.ones(n, dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None

    def _advance_path(self, path, position, bit, llr_val):
        if bit != (0 if llr_val >= 0 else 1):
            path.pm += abs(llr_val)

        path.intermediate_bits[-1][position] = bit
        for i in range(self.n - 1, -1, -1):
            path.intermediate_bits[i] = _compute_encoding_step(
                i, self.n, path.intermediate_bits[i + 1]
            )
        path.u_hat[position] = bit
        path.previous_state = _position_state(position, self.n)

    def _copy_path(self, src):
        dst = _Path(self.N, self.n, src.intermediate_llr[0])
        dst.pm = src.pm
        dst.intermediate_llr = [arr.copy() for arr in src.intermediate_llr]
        dst.intermediate_bits = [arr.copy() for arr in src.intermediate_bits]
        dst.previous_state = src.previous_state.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for position in range(self.N):
            candidates = []
            for path in paths:
                current_state = _position_state(position, self.n)
                for i in range(1, self.n + 1):
                    llr = path.intermediate_llr[i - 1]
                    if current_state[i - 1] == path.previous_state[i - 1]:
                        continue
                    if current_state[i - 1] == 0:
                        path.intermediate_llr[i] = _compute_left_alpha(llr)
                    else:
                        end = position
                        start = end - (1 << (self.n - i))
                        left_bits = path.intermediate_bits[i][start:end]
                        path.intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

                llr_val = float(path.intermediate_llr[-1][0])
                branch_bits = (0,) if self.frozen_bits[position] else (0, 1)
                for bit in branch_bits:
                    new_path = self._copy_path(path)
                    self._advance_path(new_path, position, bit, llr_val)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = (
                    p.u_hat[self.info_indices]
                    if self.info_indices is not None
                    else p.u_hat[: self.crc_length + 32]
                )
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
