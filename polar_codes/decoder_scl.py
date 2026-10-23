"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode_recursive

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
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """
    SCL 译码器。
    L=1 与 SC 完全一致；L>1 时在 SC 结果上生成单比特翻转候选并保留路径度量最小的 L 条。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u0 = sc_decode_recursive(llr_ch, self.frozen_bits)

        if self.list_size == 1:
            return u0, 0.0

        candidates = [(0.0, u0)]
        for i, idx in enumerate(self.info_idx):
            if len(candidates) >= self.list_size:
                break
            u_alt = u0.copy()
            u_alt[idx] ^= 1
            llr_bit = llr_ch[idx]
            pm = abs(llr_bit) if llr_bit != 0 else 1.0
            candidates.append((pm, u_alt))

        while len(candidates) < self.list_size:
            candidates.append((1e3, u0.copy()))

        candidates = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u, self.crc_length)]
            pm, u_hat = (
                min(valid, key=lambda x: x[0]) if valid else min(candidates, key=lambda x: x[0])
            )
        else:
            pm, u_hat = min(candidates, key=lambda x: x[0])

        return u_hat.astype(int), float(pm)
