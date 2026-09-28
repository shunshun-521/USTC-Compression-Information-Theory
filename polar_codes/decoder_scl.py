"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation
from decoder_sc import _update_llrs, _update_bits


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径级 Lazy Copy：分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_nat):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_nat
        return {"L": L, "B": B, "pm": 0.0}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = np.empty(self.N, dtype=np.float64)
        llr_nat[self.br] = llr_ch

        paths = [self._new_path(llr_nat)]

        for phi in range(self.N):
            l = int(self.br[phi])
            candidates = []

            for p in paths:
                _update_llrs(p["L"], p["B"], l, self.n, self.N)
                llr = p["L"][l, self.n]

                if l in self.frozen_idx:
                    pen = 0.0 if llr >= 0 else abs(llr)
                    candidates.append((p["pm"] + pen, p, 0))
                else:
                    pen0 = 0.0 if llr >= 0 else abs(llr)
                    pen1 = 0.0 if llr < 0 else abs(llr)
                    candidates.append((p["pm"] + pen0, p, 0))
                    candidates.append((p["pm"] + pen1, p, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                }
                child["B"][l, self.n] = 0 if l in self.frozen_idx else bit
                _update_bits(child["B"], l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        best_pm = paths[0]["pm"]
        best_u = paths[0]["B"][:, self.n].astype(int)
        crc_paths = []

        for p in paths:
            u_hat = p["B"][:, self.n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    crc_paths.append((p["pm"], u_hat))
            if p["pm"] < best_pm:
                best_pm = p["pm"]
                best_u = u_hat

        if crc_paths:
            crc_paths.sort(key=lambda x: x[0])
            best_u = crc_paths[0][1]
            best_pm = crc_paths[0][0]

        return best_u, best_pm


def verify_scl_equals_sc(N=64, frozen_bits=None, num_trials=30):
    from construction import ga_construction
    from decoder_sc import sc_decode

    if frozen_bits is None:
        info_idx, _, _ = ga_construction(N, N // 2, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

    rng = np.random.default_rng(2)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(num_trials):
        llr = rng.normal(0, 3, size=N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
