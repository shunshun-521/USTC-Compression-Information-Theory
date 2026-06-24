"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
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
    """检验 bits 是否包含正确的 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _init_path(self, llr_internal):
        path = {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "L_refs": None,
            "B_refs": None,
        }
        path["L"][:, 0] = llr_internal
        return path

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L_refs"].copy(),
            "B": path["B_refs"].copy(),
            "L_refs": None,
            "B_refs": None,
        }

    def _finalize_refs(self, path):
        if path["L_refs"] is None:
            path["L_refs"] = path["L"]
            path["B_refs"] = path["B"]

    def _path_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.br]

        paths = [self._init_path(llr_internal)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._finalize_refs(path)
                _update_llrs(path["L_refs"], path["B_refs"], l, self.n)
                llr_val = path["L_refs"][l, self.n]

                if self.frozen_bits[l]:
                    child = self._clone_path(path)
                    self._finalize_refs(child)
                    child["pm"] += self._path_penalty(llr_val, 0)
                    child["B_refs"][l, self.n] = 0
                    _update_bits(child["B_refs"], l, self.n)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = self._clone_path(path)
                        self._finalize_refs(child)
                        child["pm"] += self._path_penalty(llr_val, u_bit)
                        child["B_refs"][l, self.n] = u_bit
                        _update_bits(child["B_refs"], l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                self._finalize_refs(path)
                u_hat = path["B"][:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        u_hat = best["B"][:, self.n].astype(int)
        return u_hat, best["pm"]
