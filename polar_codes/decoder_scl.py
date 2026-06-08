"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode, sc_llr_at_phi


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_penalty(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（按比特顺序路径扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        known = np.full(self.N, np.nan)
        paths = [(known.copy(), 0.0)]

        for phi in range(self.N):
            expanded = []
            for prefix, pm in paths:
                llr_phi = sc_llr_at_phi(llr_ch, self.frozen_bits, prefix, phi)
                branch = [0] if self.frozen_bits[phi] else [0, 1]
                for bit in branch:
                    new_prefix = prefix.copy()
                    new_prefix[phi] = bit
                    expanded.append(
                        (new_prefix, pm + _path_penalty(llr_phi, bit))
                    )
            expanded.sort(key=lambda item: item[1])
            paths = expanded[: self.list_size]

        candidates = []
        for prefix, pm in paths:
            u_hat = sc_decode(llr_ch, self.frozen_bits, prefix)
            candidates.append((u_hat, pm))

        if self.crc_length > 0:
            valid = [
                (u, pm) for u, pm in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                return min(valid, key=lambda x: x[1])

        return min(candidates, key=lambda x: x[1])
