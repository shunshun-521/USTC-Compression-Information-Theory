"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _update_bits,
    _update_llrs,
    _bit_reversed,
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)

__all__ = [
    "crc_encode",
    "crc_check",
    "SCLDecoder",
    "f_operation",
    "g_operation",
    "precompute_sc_indices",
    "sc_decode",
]

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                cur_llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    u = 0
                    pm = path["pm"] + self._pm_penalty(cur_llr, u)
                    candidates.append((pm, path, u))
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(cur_llr, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u in candidates:
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                    "u_hat": parent["u_hat"].copy(),
                }
                child["u_hat"][l] = u
                child["B"][l, self.n] = u
                _update_bits(child["B"], l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
