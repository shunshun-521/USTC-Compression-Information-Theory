"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from polar_scd import PolarSCD
from polar_scd_utils import bit_reversed, hard_decision


def _poly_bits(crc_length):
    poly = 0x07 if crc_length == 8 else 0x8005
    return np.array([int(x) for x in format(poly, f"0{crc_length}b")], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = np.zeros(crc_length, dtype=np.int8)
    pb = _poly_bits(crc_length)
    for bit in info_bits:
        fb = reg[0] ^ bit
        if fb:
            reg[:-1] = reg[1:] ^ pb[: crc_length - 1]
            reg[-1] = pb[-1]
        else:
            reg[:-1] = reg[1:]
            reg[-1] = 0
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    bits = np.asarray(bits, dtype=np.int8)
    reg = np.zeros(crc_length, dtype=np.int8)
    pb = _poly_bits(crc_length)
    for bit in bits:
        fb = reg[0] ^ bit
        if fb:
            reg[:-1] = reg[1:] ^ pb[: crc_length - 1]
            reg[-1] = pb[-1]
        else:
            reg[:-1] = reg[1:]
            reg[-1] = 0
    return np.all(reg == 0)


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm(self, pm, llr, u):
        v = hard_decision(llr)
        return pm if u == v else pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{"pm": 0.0, "eng": PolarSCD(N, llr_ch), "u": np.zeros(N, dtype=np.int8)}]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                eng = path["eng"]
                eng.update_llrs(l)
                llr = eng.L[l, n]
                if l in self.frozen_set:
                    eng.B[l, n] = 0
                    eng.update_bits(l)
                    path["u"][l] = 0
                    new_paths.append(
                        {
                            "pm": self._pm(path["pm"], llr, 0),
                            "eng": eng,
                            "u": path["u"].copy(),
                        }
                    )
                else:
                    for u in (0, 1):
                        eng_c = copy.deepcopy(path["eng"])
                        eng_c.B[l, n] = u
                        eng_c.update_bits(l)
                        u_hat = path["u"].copy()
                        u_hat[l] = u
                        new_paths.append(
                            {
                                "pm": self._pm(path["pm"], llr, u),
                                "eng": eng_c,
                                "u": u_hat,
                            }
                        )
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = [(p["pm"], p["u"]) for p in paths]
        if self.crc_length > 0:
            ok = []
            for pm, u in candidates:
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    ok.append((pm, u))
            if ok:
                ok.sort(key=lambda x: x[0])
                return ok[0][1], ok[0][0]
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
