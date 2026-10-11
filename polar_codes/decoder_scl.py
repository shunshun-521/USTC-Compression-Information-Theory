"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _SCDCore, sc_decode
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_encode_bits(info_bits, width, poly):
    top = 1 << width
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & (top - 1)
            else:
                reg = (reg << 1) & (top - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        reg = _crc_encode_bits(info_bits, 8, CRC8_POLY)
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = _crc_encode_bits(info_bits, 16, CRC16_POLY)
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（基于与 SC 相同的因子图更新）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = []
        core = _SCDCore(self.N, self.frozen_bits)
        core.L[:, 0] = llr_ch
        paths.append({"core": core, "pm": 0.0, "u": np.zeros(self.N, dtype=int)})

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                path["core"]._update_llrs(l)
                llr_bit = path["core"].L[l, self.n]
                if l in self.frozen:
                    pm = path["pm"] + self._pm_penalty(llr_bit, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr_bit, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, u_bit in candidates[: self.L]:
                core = _SCDCore(self.N, self.frozen_bits)
                core.L = parent["core"].L.copy()
                core.B = parent["core"].B.copy()
                core.B[l, self.n] = u_bit
                core._update_bits(l)
                u = parent["u"].copy()
                u[l] = u_bit
                new_paths.append({"core": core, "pm": pm, "u": u})
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_idx], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
