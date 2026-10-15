"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from polar_utils import bit_reversed, hard_decision
from decoder_sc import _SCDState


def _crc_remainder(bits, crc_length):
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    return _crc_remainder(np.asarray(bits, dtype=int), crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"state": _SCDState(self.N, self.n), "pm": 0.0, "phi": 0}]
        paths[0]["state"].set_channel(llr_ch)

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                st = path["state"]
                st.update_llrs(l)
                llr = st.L[l, self.n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    st2 = copy.deepcopy(st)
                    st2.B[l, self.n] = 0
                    st2.update_bits(l)
                    new_paths.append({"state": st2, "pm": pm, "phi": i + 1})
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, u)
                        st2 = copy.deepcopy(st)
                        st2.B[l, self.n] = u
                        st2.update_bits(l)
                        new_paths.append({"state": st2, "pm": pm, "phi": i + 1})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            info_positions = np.where(self.frozen_bits == 0)[0]
            for p in paths:
                u_hat = p["state"].B[:, self.n].astype(int)
                if crc_check(u_hat[info_positions], self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["state"].B[:, self.n].astype(int), best["pm"]
