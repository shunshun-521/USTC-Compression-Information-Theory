"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg == 0


def _scl_recursive(llr, frozen, list_size):
    """返回 [(pm, u_hat, u_hat_up)]，与 SC 递归结构一致。"""
    n = len(llr)

    if n == 1:
        cands = []
        if frozen[0]:
            pm = 0.0 if llr[0] >= 0 else abs(llr[0])
            u = np.array([0], dtype=int)
            cands.append((pm, u, u.astype(np.float64)))
        else:
            hard = 0 if llr[0] >= 0 else 1
            for u_bit in (0, 1):
                pm = 0.0 if u_bit == hard else abs(llr[0])
                u = np.array([u_bit], dtype=int)
                cands.append((pm, u, u.astype(np.float64)))
        cands.sort(key=lambda x: x[0])
        return cands[:list_size]

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    fr1, fr2 = frozen[:half], frozen[half:]

    llr_up = f_operation(llr1, llr2)
    upper_paths = _scl_recursive(llr_up, fr1, list_size)

    merged = []
    for pm_u, u1, u1_up in upper_paths:
        llr_low = g_operation(llr1, llr2, u1_up)
        lower_paths = _scl_recursive(llr_low, fr2, list_size)
        for pm_l, u2, u2_up in lower_paths:
            u_hat = np.concatenate([u1, u2])
            u1_up_xor = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
            u_hat_up = np.concatenate([u1_up_xor, u2_up])
            merged.append((pm_u + pm_l, u_hat, u_hat_up))

    merged.sort(key=lambda x: x[0])
    return merged[:list_size]


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _scl_recursive(llr_ch, self.frozen_bits, self.list_size)
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            for pm, u, _ in paths:
                if crc_check(u[self.info_indices], self.crc_length):
                    return u.astype(int), float(pm)

        pm_best, u_best, _ = paths[0]
        return u_best.astype(int), float(pm_best)
