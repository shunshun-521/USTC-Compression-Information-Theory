"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_bit_llr, sc_decode_recursive


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.asarray(bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


def _bit_llr(llr_ch, frozen_bits, u_prefix, phi):
    return sc_bit_llr(llr_ch, frozen_bits, u_prefix, phi)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen_bits.astype(int))
            return u, 0.0

        paths = [(0.0, np.zeros(self.N, dtype=int))]
        frozen_int = self.frozen_bits.astype(int)

        for phi in range(self.N):
            new_paths = []
            for pm, u_hat in paths:
                llr_bit = _bit_llr(llr_ch, frozen_int, u_hat, phi)
                if self.frozen_bits[phi]:
                    u2 = u_hat.copy()
                    u2[phi] = 0
                    new_paths.append((pm + self._pm_penalty(llr_bit, 0), u2))
                else:
                    for bit in (0, 1):
                        u2 = u_hat.copy()
                        u2[phi] = bit
                        new_paths.append((pm + self._pm_penalty(llr_bit, bit), u2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in paths if crc_check(u, self.crc_length)]
            pm, u_hat = min(valid if valid else paths, key=lambda x: x[0])
        else:
            pm, u_hat = min(paths, key=lambda x: x[0])

        return u_hat, pm
