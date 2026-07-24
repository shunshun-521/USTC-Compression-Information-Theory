"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _scl_decode_tree(llr, frozen_bits, list_size):
    """递归 SCL 译码树"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n > 1:
        half = n // 2
        l1, l2 = llr[:half], llr[half:]
        f1, f2 = frozen_bits[:half], frozen_bits[half:]

        left_paths = _scl_decode_tree(f_operation(l1, l2), f1, list_size)
        all_paths = []
        for u1, pm1, u1_up in left_paths:
            llr_lower = g_operation(l1, l2, u1_up)
            lower_paths = _scl_decode_tree(llr_lower, f2, list_size)
            for u2, pm2, u2_up in lower_paths:
                u_hat = np.concatenate([u1, u2])
                pm = pm1 + pm2
                u1_up_new = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
                u_up = np.concatenate([u1_up_new, u2_up])
                all_paths.append((u_hat, pm, u_up))

        all_paths.sort(key=lambda x: x[1])
        return all_paths[:list_size]

    if frozen_bits[0]:
        return [(np.array([0], dtype=int), 0.0, np.array([0.0]))]

    paths = []
    for bit in (0, 1):
        penalty = 0.0 if (bit == 0 and llr[0] >= 0) or (bit == 1 and llr[0] < 0) else abs(llr[0])
        paths.append((np.array([bit], dtype=int), penalty, np.array([float(bit)])))
    paths.sort(key=lambda x: x[1])
    return paths[:list_size]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        paths = _scl_decode_tree(llr_ch, self.frozen_bits, self.list_size)
        if self.crc_length > 0:
            valid = []
            for u_hat, pm, _ in paths:
                info = u_hat[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append((u_hat, pm))
            if valid:
                u_hat, pm = min(valid, key=lambda x: x[1])
                return u_hat, pm
        u_hat, pm, _ = paths[0]
        return u_hat, pm
