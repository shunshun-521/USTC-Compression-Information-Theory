"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llr,
    _update_bits,
    _update_llrs,
    sc_decode,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _bits_to_bytes_msb(bits):
    """将比特流按 MSB 优先打包为字节（末字节不足 8 位时右侧补 0）。"""
    bits = list(bits)
    nbytes = (len(bits) + 7) // 8
    out = bytearray()
    for i in range(nbytes):
        val = 0
        for j in range(8):
            idx = i * 8 + j
            val = (val << 1) | (int(bits[idx]) if idx < len(bits) else 0)
        out.append(val)
    return bytes(out)


def _crc8_byte(data):
    """标准 CRC-8（多项式 0x07）按字节更新。"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(data):
    """标准 CRC-16（多项式 0x8005）按字节更新。"""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    msg_bytes = _bits_to_bytes_msb(info_bits)
    if crc_length == 8:
        rem = _crc8_byte(msg_bytes)
    else:
        rem = _crc16_byte(msg_bytes)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    msg_bytes = _bits_to_bytes_msb(bits)
    if crc_length == 8:
        return _crc8_byte(msg_bytes) == 0
    return _crc16_byte(msg_bytes) == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length

        if self.list_size == 1 and crc_length == 0:
            self._use_sc = True
        else:
            self._use_sc = False

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        if self._use_sc:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr0 = _prepare_channel_llr(llr_ch)
        paths = [{
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "u_hat": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr0

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    u = 0
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    new_path = self._clone_path(path)
                    new_path["pm"] += penalty
                    new_path["u_hat"][l] = u
                    new_path["B"][l, self.n] = u
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        hard = 0 if llr >= 0 else 1
                        penalty = 0.0 if u == hard else abs(llr)
                        new_path = self._clone_path(path)
                        new_path["pm"] += penalty
                        new_path["u_hat"][l] = u
                        new_path["B"][l, self.n] = u
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        return best["u_hat"].astype(int), best["pm"]

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            info_idx = np.where(info_mask)[0]
            valid = []
            for path in paths:
                info_bits = path["u_hat"][info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                return min(valid, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
