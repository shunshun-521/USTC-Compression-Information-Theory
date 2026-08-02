"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    LLR_MAX,
    _polar_decode_sc_recursive,
    sc_decode_recursive,
)


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
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
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
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, u):
    llr_clip = np.clip(llr, -LLR_MAX, LLR_MAX)
    return pm + np.log1p(np.exp(-(1 - 2 * u) * llr_clip))


def _scl_decode_recursive(llr_ch, frozen_ind, list_size):
    """递归 SCL 译码核心。"""
    n = len(frozen_ind)

    if n == 1:
        cur_llr = llr_ch[0]
        if frozen_ind[0]:
            return [(np.array([0.0]), np.array([0.0]), 0.0)]
        return [
            (np.array([0.0]), np.array([0.0]), _pm_update(0.0, cur_llr, 0)),
            (np.array([1.0]), np.array([1.0]), _pm_update(0.0, cur_llr, 1)),
        ]

    half = n // 2
    llr1, llr2 = llr_ch[:half], llr_ch[half:]
    f1, f2 = frozen_ind[:half], frozen_ind[half:]

    x_llr1 = f_operation(llr1, llr2)
    left_paths = _scl_decode_recursive(x_llr1, f1, list_size)

    all_paths = []
    for u1, u1_up, pm1 in left_paths:
        x_llr2 = g_operation(llr1, llr2, u1_up)
        right_paths = _scl_decode_recursive(x_llr2, f2, list_size)
        for u2, u2_up, pm2 in right_paths:
            u_hat = np.concatenate([u1, u2]).astype(int)
            u1_enc = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
            u_up = np.concatenate([u1_enc, u2_up])
            all_paths.append((u_hat, u_up, pm1 + pm2))

    all_paths.sort(key=lambda x: x[2])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        frozen_ind = self.frozen_bits.astype(float)
        paths = _scl_decode_recursive(
            np.asarray(llr_ch, dtype=np.float64),
            frozen_ind,
            self.list_size,
        )

        if self.crc_length > 0:
            for u_hat, _, pm in sorted(paths, key=lambda x: x[2]):
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, pm

        best = min(paths, key=lambda x: x[2])
        return best[0], best[2]
