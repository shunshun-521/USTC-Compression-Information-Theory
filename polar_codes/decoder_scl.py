"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _SCDState, _bit_reverse, sc_decode

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_state(self, state):
        new_state = _SCDState(state.llr_ch, self.frozen_bits)
        new_state.L = state.L.copy()
        new_state.B = state.B.copy()
        return new_state

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"state": _SCDState(llr_ch, self.frozen_bits), "pm": 0.0}]

        for i in range(self.N):
            l = _bit_reverse(i, self.n)
            candidates = []

            for path in paths:
                st = path["state"]
                st.update_llrs(l)
                llr = st.L[l, self.n]

                if self.frozen_bits[l]:
                    new_st = self._clone_state(st)
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    new_st.B[l, self.n] = 0
                    new_st.update_bits(l)
                    candidates.append({"state": new_st, "pm": pm})
                else:
                    for u_bit in (0, 1):
                        new_st = self._clone_state(st)
                        pm = path["pm"] + self._pm_penalty(llr, u_bit)
                        new_st.B[l, self.n] = u_bit
                        new_st.update_bits(l)
                        candidates.append({"state": new_st, "pm": pm})

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["state"].B[:, self.n].astype(np.int32)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda x: x["pm"])
        u_hat = best["state"].B[:, self.n].astype(np.int32)
        return u_hat, best["pm"]
