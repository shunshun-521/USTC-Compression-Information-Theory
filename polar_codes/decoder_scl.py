"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _SCDEngine, bit_reversed, sc_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _bits_to_bytes_msb(bits):
    bits = np.asarray(bits, dtype=int).ravel()
    n = len(bits)
    nbytes = (n + 7) // 8
    padded = np.zeros(nbytes * 8, dtype=int)
    padded[:n] = bits
    out = bytearray()
    for i in range(0, len(padded), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | padded[i + j]
        out.append(b)
    return out


def _crc_on_bytes(data_bytes, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for byte in data_bytes:
        reg ^= byte << (crc_length - 8) if crc_length >= 8 else byte
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    data_bytes = _bits_to_bytes_msb(info_bits)
    rem = _crc_on_bytes(data_bytes, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    tx_crc = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], tx_crc)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int) if info_indices is not None else None
        )

    @staticmethod
    def _penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = []
        eng0 = _SCDEngine(self.N, self.frozen_set)
        eng0.set_channel(llr_ch)
        paths.append({"engine": eng0, "pm": 0.0, "u": np.zeros(self.N, dtype=int)})

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []
            for path in paths:
                eng = path["engine"]
                eng.update_llrs(l)
                llr0 = eng.L[l, self.n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._penalty(llr0, 0)
                    eng2 = self._clone_engine(eng)
                    eng2.B[l, self.n] = 0
                    eng2.update_bits(l)
                    u = path["u"].copy()
                    u[l] = 0
                    new_paths.append({"engine": eng2, "pm": pm, "u": u})
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._penalty(llr0, u_bit)
                        eng2 = self._clone_engine(eng)
                        eng2.B[l, self.n] = u_bit
                        eng2.update_bits(l)
                        u = path["u"].copy()
                        u[l] = u_bit
                        new_paths.append({"engine": eng2, "pm": pm, "u": u})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                if self.info_indices is not None:
                    bits = p["u"][self.info_indices]
                else:
                    bits = p["u"]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]
        return best["u"], best["pm"]

    @staticmethod
    def _clone_engine(eng):
        new = _SCDEngine(eng.N, eng.frozen)
        new.L = eng.L.copy()
        new.B = eng.B.copy()
        return new
