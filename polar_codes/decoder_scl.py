"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import polar_encode, polar_generator_matrix
from utils_gf2 import get_ginv, gf2_decode_from_llr
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC（bits 为信息+CRC 串联）。"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _path_metric(llr, x):
    """对数域相关度量（越大越好）。"""
    llr = np.asarray(llr, dtype=np.float64)
    x = np.asarray(x, dtype=int)
    s = 1 - 2 * x
    return float(np.sum(llr * s))


class SCLDecoder:
    """SCL 译码器（基于码字硬判决候选 + GF(2) 求逆，列表裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_indices = info_indices
        G = polar_generator_matrix(N)
        self._g_inv = get_ginv(N, G)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        x0 = (llr < 0).astype(int)
        candidates = {tuple(x0)}

        order = np.argsort(np.abs(llr))
        for idx in order:
            if len(candidates) >= self.list_size * 4:
                break
            new_cands = set()
            for x_t in candidates:
                x = np.array(x_t, dtype=int)
                x_flip = x.copy()
                x_flip[idx] ^= 1
                new_cands.add(tuple(x))
                new_cands.add(tuple(x_flip))
            candidates = new_cands

        paths = []
        for x_t in candidates:
            x = np.array(x_t, dtype=int)
            u = (x @ self._g_inv) % 2
            u[self.frozen] = 0
            pm = -_path_metric(llr, x)
            paths.append((pm, u, x))

        paths.sort(key=lambda t: t[0])

        if self.crc_length > 0 and self.info_indices is not None:
            valid = []
            for pm, u, _ in paths:
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, u))
            if valid:
                paths = [(pm, u, None) for pm, u in valid]

        best_pm, best_u, _ = paths[0]
        return best_u, best_pm
