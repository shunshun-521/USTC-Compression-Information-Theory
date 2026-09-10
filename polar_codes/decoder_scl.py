"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _cn_op, _vn_op, _hard_bit, sc_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_division(data_bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for bit in data_bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & top:
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & top:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_division(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _metric_penalty(llr, u):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


def _leaf_paths(llr, frozen_val, list_size):
    if frozen_val == 1:
        return [(np.array([0.0]), np.array([0.0]), 0.0)]
    paths = []
    for b in (0, 1):
        u = np.array([float(b)])
        paths.append((u, u, _metric_penalty(llr, b)))
    paths.sort(key=lambda x: x[2])
    return paths[:list_size]


def _polar_scl_recursive(llr_ch, frozen_ind, list_size):
    n = len(llr_ch)
    if n == 1:
        return _leaf_paths(llr_ch[0], frozen_ind[0], list_size)

    half = n // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    f1 = frozen_ind[:half]
    f2 = frozen_ind[half:]

    x1 = _cn_op(llr1, llr2)
    left_paths = _polar_scl_recursive(x1, f1, list_size)

    all_paths = []
    for u1, up1, pm1 in left_paths:
        x2 = _vn_op(llr1, llr2, up1)
        right_paths = _polar_scl_recursive(x2, f2, list_size)
        for u2, up2, pm2 in right_paths:
            u = np.concatenate([u1, u2])
            up_left = (up1.astype(np.int8) ^ up2.astype(np.int8)).astype(np.float64)
            up = np.concatenate([up_left, up2])
            all_paths.append((u, up, pm1 + pm2))

    all_paths.sort(key=lambda x: x[2])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器（递归列表，Lazy Copy 通过路径裁剪实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_ind = self.frozen_bits.astype(np.float64)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = _polar_scl_recursive(llr, self.frozen_ind, self.list_size)

        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, _, pm in paths
                if crc_check(u.astype(np.int8)[self.info_indices], self.crc_length)
            ]
            if valid:
                best_u, best_pm = min(valid, key=lambda x: x[1])
            else:
                best_u, best_pm = paths[0][0], paths[0][2]
        else:
            best_u, best_pm = paths[0][0], paths[0][2]

        return best_u.astype(np.int8), best_pm
