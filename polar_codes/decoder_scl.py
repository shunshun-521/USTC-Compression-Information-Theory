"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    sc_decode_recursive,
    f_operation,
    g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(
        bits[-crc_length:],
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
    )


def _pm_bit(pm, llr, u):
    if (llr >= 0 and u == 0) or (llr < 0 and u == 1):
        return pm
    return pm + abs(llr)


def _scl_recursive(llr_node, frozen_node, list_size):
    """递归 SCL，返回路径列表 [(u_hat, pm, u_hat_up), ...]。"""
    n = len(llr_node)
    if n == 1:
        paths = []
        if frozen_node[0]:
            pm = abs(llr_node[0]) if llr_node[0] < 0 else 0.0
            paths.append((np.array([0], dtype=int), pm, np.array([0.0])))
        else:
            for u in (0, 1):
                pm = _pm_bit(0.0, llr_node[0], u)
                paths.append((np.array([u], dtype=int), pm, np.array([float(u)])))
        paths.sort(key=lambda x: x[1])
        return paths[:list_size]

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    frozen1 = frozen_node[:half]
    frozen2 = frozen_node[half:]

    paths_up = _scl_recursive(f_operation(llr1, llr2), frozen1, list_size)
    merged = []

    for u1, pm1, u1_up in paths_up:
        llr2_in = g_operation(llr1, llr2, u1_up)
        paths_low = _scl_recursive(llr2_in, frozen2, list_size)
        for u2, pm2, u2_up in paths_low:
            u_hat = np.concatenate([u1, u2])
            u_up_left = np.bitwise_xor(
                u1_up.astype(np.int64), u2_up.astype(np.int64)
            ).astype(np.float64)
            u_hat_up = np.concatenate([u_up_left, u2_up])
            merged.append((u_hat, pm1 + pm2, u_hat_up))

    merged.sort(key=lambda x: x[1])
    return merged[:list_size]


class SCLDecoder:
    """SCL 译码器（递归列表 + CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.L == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _scl_recursive(llr_ch, self.frozen_bits, self.L)

        if self.crc_length > 0:
            valid = [
                (u, pm, _) for u, pm, _ in paths
                if crc_check(u[self.info_idx], self.crc_length)
            ]
            if valid:
                paths = valid

        u_hat, pm, _ = min(paths, key=lambda x: x[1])
        return u_hat, pm
