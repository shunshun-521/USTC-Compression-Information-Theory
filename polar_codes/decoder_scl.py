"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    )


def _pm_update(pm, llr, u):
    v = 0 if llr >= 0 else 1
    if u != v:
        pm += abs(llr)
    return pm


def _scl_decode_recursive(llr, frozen, list_size):
    """递归 SCL，返回 [(pm, u_hat, u_hat_up), ...]"""
    frozen = np.asarray(frozen, dtype=bool)
    n = len(llr)

    if n == 1:
        out = []
        if frozen[0]:
            out.append((0.0, np.array([0.0]), np.array([0.0])))
        else:
            for u in (0.0, 1.0):
                pm = _pm_update(0.0, llr[0], int(u))
                out.append((pm, np.array([u]), np.array([u])))
        out.sort(key=lambda x: x[0])
        return out[:list_size]

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    fr1, fr2 = frozen[:half], frozen[half:]

    paths_left = _scl_decode_recursive(f_operation(llr1, llr2), fr1, list_size)

    all_paths = []
    for pm, u_hat1, u_hat1_up in paths_left:
        llr_right = g_operation(llr1, llr2, u_hat1_up)
        paths_right = _scl_decode_recursive(llr_right, fr2, list_size)
        for pm2, u_hat2, u_hat2_up in paths_right:
            pm_total = pm + pm2
            u_hat = np.concatenate([u_hat1, u_hat2])
            u1c = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(np.float64)
            u_hat_up = np.concatenate([u1c, u_hat2_up])
            all_paths.append((pm_total, u_hat, u_hat_up))

    all_paths.sort(key=lambda x: x[0])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _scl_decode_recursive(llr_ch, self.frozen_bits, self.list_size)
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u, _ in paths
                if crc_check(u[self.info_indices].astype(int), self.crc_length)
            ]
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat.astype(int), pm

        pm, u_hat, _ = paths[0]
        return u_hat.astype(int), pm
