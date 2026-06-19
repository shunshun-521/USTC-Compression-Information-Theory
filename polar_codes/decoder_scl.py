"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr_exact,
    _update_bits,
    _update_llrs,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    extract = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & mask
    if extract:
        reg ^= poly
    return reg


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    crc_bits = []
    for _ in range(crc_length):
        crc_bits.append((reg >> (crc_length - 1)) & 1)
        reg = _crc_update(reg, 0, poly, crc_length)
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = {i for i in range(N) if self.frozen_bits[i]}
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(self.frozen_bits == 0)[0]
        )
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _new_path(self, parent, l, bit_val, pm_penalty):
        child = {
            "L": parent["L"].copy(),
            "B": parent["B"].copy(),
            "u_hat": parent["u_hat"].copy(),
            "pm": parent["pm"] + pm_penalty,
        }
        child["u_hat"][l] = bit_val
        child["B"][l, self.n] = bit_val
        _update_bits(child["B"], l, self.n, self.N)
        return child

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=int),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    candidates.append(self._new_path(path, l, 0, self._pm_penalty(llr, 0)))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            self._new_path(path, l, bit, self._pm_penalty(llr, bit))
                        )
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path["u_hat"]
            pm = path["pm"]
            if self.crc_length > 0:
                info_sorted = np.sort(self.info_indices)
                info_bits = u_hat[info_sorted]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[1]:
                        best_crc = (u_hat, pm)
            if best_pm is None or pm < best_pm[1]:
                best_pm = (u_hat, pm)

        if best_crc is not None:
            return best_crc
        return best_pm
