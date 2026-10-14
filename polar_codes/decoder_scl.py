"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _PolarTreeSC

# ==================== CRC 工具 ====================

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
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class SCLDecoder:
    """SCL / CA-SCL 译码器（极化树 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        decoder = _PolarTreeSC(self.frozen_bits, list_size=self.list_size)
        u_hat = decoder.decode(llr_ch)
        pm = decoder.metrics[0] if decoder.metrics else 0.0

        if self.crc_length > 0 and self.list_size > 1:
            # 重新运行以收集多路径（L>1 时 _PolarTreeSC 已裁剪为最优路径）
            decoder = _PolarTreeSC(self.frozen_bits, list_size=self.list_size)
            decoder.decode(llr_ch)
            for idx in range(len(decoder.decisions)):
                info_bits = decoder.decisions[idx][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return decoder.decisions[idx].astype(int).copy(), decoder.metrics[idx]
        return u_hat, pm


def verify_scl_equals_sc(N=64, frozen_bits=None, llr=None):
    from construction import ga_construction
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if llr is None:
        rng = np.random.default_rng(1)
        llr = rng.normal(0, 2, size=N)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
        raise AssertionError("SCL L=1 != SC")
