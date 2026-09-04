"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _SCDCore,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
    _map_channel_llrs,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    r = crc_length

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (r - 1)
        for _ in range(8):
            top = (reg >> (r - 1)) & 1 if r == 16 else (reg & 0x80)
            if r == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if top:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF

    if crc_length == 8:
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（基于 Vangala 置换 SC 扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        core = _SCDCore(self.N, self.frozen_bits)
        core.set_llr(llr_ch)
        return {"core": core, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        core = _SCDCore(self.N, self.frozen_bits)
        core.L = path["core"].L.copy()
        core.B = path["core"].B.copy()
        return {"core": core, "pm": path["pm"], "u_hat": path["u_hat"].copy()}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                core = path["core"]
                core._update_llrs(l)
                llr0 = core.L[l, core.n]

                if l in self.frozen:
                    child = self._copy_path(path)
                    child["pm"] += _pm_penalty(llr0, 0)
                    child["core"].B[l, core.n] = 0
                    child["u_hat"][l] = 0
                    child["core"]._update_bits(l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = self._copy_path(path)
                        child["pm"] += _pm_penalty(llr0, u)
                        child["core"].B[l, core.n] = u
                        child["u_hat"][l] = u
                        child["core"]._update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
