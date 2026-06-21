"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _leaf_llr, _path_metric_update, _permute_llr, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_perm = _permute_llr(llr_ch, self.N)
        paths = [{"pm": 0.0, "prefix": {}}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = _leaf_llr(
                    llr_perm, self.frozen_bits, path["prefix"], phi
                )
                if self.frozen_bits[phi]:
                    prefix = dict(path["prefix"])
                    prefix[phi] = 0
                    candidates.append(
                        (_path_metric_update(path["pm"], llr, 0), prefix)
                    )
                else:
                    for bit in (0, 1):
                        prefix = dict(path["prefix"])
                        prefix[phi] = bit
                        pm = _path_metric_update(path["pm"], llr, bit)
                        candidates.append((pm, prefix))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]
            paths = [{"pm": c[0], "prefix": c[1]} for c in candidates]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = _finalize_path(llr_perm, self.frozen_bits, p["prefix"])
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append((p["pm"], u))
            if valid:
                best = min(valid, key=lambda x: x[0])
                return best[1], best[0]

        best = min(paths, key=lambda x: x["pm"])
        u_hat = _finalize_path(llr_perm, self.frozen_bits, best["prefix"])
        return u_hat, best["pm"]


def _finalize_path(llr_perm, frozen_bits, prefix):
    from decoder_sc import _sc_tree_decode

    return _sc_tree_decode(llr_perm, frozen_bits, prefix=prefix)
