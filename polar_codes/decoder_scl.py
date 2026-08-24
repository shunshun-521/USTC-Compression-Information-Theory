"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import cn_operation, g_operation, sc_decode_recursive

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in np.asarray(info_bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    return np.concatenate([info_bits, _crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _path_metric_penalty(llr, u_val):
    expected = 0 if llr >= 0 else 1
    return 0.0 if u_val == expected else abs(llr)


def _polar_decode_scl_recursive(llr_ch, frozen_ind, list_size):
    """递归 SCL 译码"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=bool)

    if n > 1:
        half = n // 2
        llr1, llr2 = llr_ch[:half], llr_ch[half:]
        frozen1, frozen2 = frozen_ind[:half], frozen_ind[half:]

        paths1 = _polar_decode_scl_recursive(cn_operation(llr1, llr2), frozen1, list_size)
        all_paths = []
        for u_hat1, u_hat1_up, pm1 in paths1:
            paths2 = _polar_decode_scl_recursive(
                g_operation(llr1, llr2, u_hat1_up), frozen2, list_size
            )
            for u_hat2, u_hat2_up, pm2 in paths2:
                u_hat = np.concatenate([u_hat1, u_hat2])
                u_hat1_up = np.bitwise_xor(u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8))
                u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
                all_paths.append((u_hat, u_hat_up, pm1 + pm2))

        all_paths.sort(key=lambda x: x[2])
        return all_paths[:list_size]

    if frozen_ind[0]:
        penalty = _path_metric_penalty(llr_ch[0], 0)
        u_hat = np.array([0], dtype=np.int8)
        return [(u_hat, u_hat.copy(), penalty)]

    paths = []
    for u_val in (0, 1):
        penalty = _path_metric_penalty(llr_ch[0], u_val)
        u_hat = np.array([u_val], dtype=np.int8)
        paths.append((u_hat, u_hat.copy(), penalty))
    paths.sort(key=lambda x: x[2])
    return paths[:list_size]


class SCLDecoder:
    """SCL 译码器（递归实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _polar_decode_scl_recursive(llr_ch, self.frozen_bits, self.list_size)

        best = None
        if self.crc_length > 0:
            for u_hat, _, pm in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or pm < best[1]:
                        best = (u_hat, pm)
        if best is None:
            u_hat, _, pm = paths[0]
            best = (u_hat, pm)

        return best[0].astype(np.int8), best[1]
