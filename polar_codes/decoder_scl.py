"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
基于 Permuted SCD 结构
"""
import numpy as np

from encoder import bit_reversed
from decoder_sc import _SCDState, _update_llrs, _update_bits
from polar_ops import active_llr_level, active_bit_level, upper_llr, lower_llr

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8 if crc_len <= 8 else 1):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径复制 + Permuted SCD）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )

    @staticmethod
    def _penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs_for_l(self, state, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    state.L[j, s + 1] = upper_llr(state.L[j, s], state.L[j + branch_size, s])
                else:
                    state.L[j, s + 1] = lower_llr(
                        state.L[j, s],
                        state.L[j - branch_size, s],
                        state.B[j - branch_size, s + 1],
                    )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"state": _SCDState(self.N, self.n, llr_ch.copy()), "pm": 0.0}]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                st = path["state"]
                self._update_llrs_for_l(st, l)
                cur_llr = st.L[l, self.n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._penalty(cur_llr, 0)
                    st.B[l, self.n] = 0
                    _update_bits(st, l)
                    new_paths.append({"state": st, "pm": pm})
                else:
                    for u in (0, 1):
                        child_st = _SCDState(self.N, self.n, np.zeros(self.N))
                        child_st.L = st.L.copy()
                        child_st.B = st.B.copy()
                        pm = path["pm"] + self._penalty(cur_llr, u)
                        child_st.B[l, self.n] = u
                        _update_bits(child_st, l)
                        new_paths.append({"state": child_st, "pm": pm})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                u_hat = p["state"].B[:, self.n].astype(int)
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_paths.append(p)

        best = min(crc_paths if crc_paths else paths, key=lambda p: p["pm"])
        u_hat = best["state"].B[:, self.n].astype(int)
        return u_hat, best["pm"]
