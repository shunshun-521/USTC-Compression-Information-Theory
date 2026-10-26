"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _scl_decode_recursive(llr_ch, frozen_ind, list_size):
    """SCL 递归核，返回路径列表 [(u_hat, pm), ...]。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=int)
    n = len(llr_ch)

    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fz1 = frozen_ind[:half]
        fz2 = frozen_ind[half:]

        llr_left = f_operation(llr1, llr2)
        paths_left = _scl_decode_recursive(llr_left, fz1, list_size)

        merged = []
        for u1, u1_up, pm1 in paths_left:
            llr_right = g_operation(llr1, llr2, u1_up)
            paths_right = _scl_decode_recursive(llr_right, fz2, list_size)
            for u2, u2_up, pm2 in paths_right:
                u_hat = np.concatenate([u1, u2])
                u1_up_new = (u1_up ^ u2_up) % 2
                u_up = np.concatenate([u1_up_new, u2_up])
                merged.append((u_hat, u_up, pm1 + pm2))

        merged.sort(key=lambda x: x[2])
        return merged[:list_size]

    # leaf
    llr = llr_ch[0]
    paths = []
    if frozen_ind[0]:
        paths.append((np.array([0], int), np.array([0], int), abs(llr) if llr < 0 else 0.0))
    else:
        hard = 0 if llr >= 0 else 1
        for u in (0, 1):
            pm = 0.0 if u == hard else abs(llr)
            paths.append((np.array([u], int), np.array([u], int), pm))
        paths.sort(key=lambda x: x[2])
        return paths[:list_size]
    return paths[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        frozen_ind = self.frozen_bits.astype(int)
        paths = _scl_decode_recursive(llr_ch, frozen_ind, self.list_size)

        crc_ok = []
        for u_hat, _, pm in paths:
            if self.crc_length > 0:
                bits = u_hat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    crc_ok.append((u_hat, pm))
        if crc_ok:
            u_hat, pm = min(crc_ok, key=lambda x: x[1])
        else:
            u_hat, _, pm = paths[0]
        return u_hat.copy(), pm
