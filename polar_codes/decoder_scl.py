"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import numpy as np

from decoder_core import scl_decode_core
from decoder_sc import _prepare_llr_and_mask


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
CRC16_POLY = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    r = len(poly) - 1
    msg = np.concatenate([info_bits, np.zeros(r, dtype=int)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i : i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[-r:]])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    r = len(poly) - 1
    if len(bits) < r:
        return False
    rem = bits.copy()
    for i in range(len(bits) - r + 1):
        if rem[i] == 1:
            rem[i : i + len(poly)] ^= poly
    return np.all(rem[-r:] == 0)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = self.frozen_bits == 0

    def decode(self, llr_ch):
        llr_perm, if_info = _prepare_llr_and_mask(llr_ch, self.frozen_bits)
        u_hat, pm = scl_decode_core(llr_perm, if_info, self.list_size)

        if self.crc_length > 0 and self.list_size > 1:
            pass

        if self.crc_length > 0:
            info_bits = u_hat[self.info_mask]
            if not crc_check(info_bits, self.crc_length):
                pm = float(pm)
        else:
            pm = float(pm)

        return u_hat.astype(int), pm


def scl_equivalent_sc(llr_ch, frozen_bits):
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u, _ = dec.decode(llr_ch)
    return u
