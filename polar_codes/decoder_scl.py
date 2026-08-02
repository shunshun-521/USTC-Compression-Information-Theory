"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _llr_to_p1,
    _cnop_p1,
    _vnop_p1,
    _p1_to_llr,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    """计算 CRC 校验位（LFSR，多项式含隐式 x^r 项）"""
    poly = (CRC8_POLY if crc_length == 8 else CRC16_POLY) | (1 << crc_length)
    reg = np.zeros(crc_length + len(info_bits) + crc_length, dtype=int)
    reg[: len(info_bits)] = info_bits
    n = len(info_bits)
    for i in range(n):
        if reg[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    reg[i + j] ^= 1
    return reg[n : n + crc_length]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


def _pm_penalty(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit == hard else abs(llr_val)


def _scl_paths_p1(y, frozen, list_size):
    """
    P1 域递归 SCL。
    每条路径返回 (u, x_partial, pm)，右子树使用 x_partial 而非硬比特。
    """
    N = len(y)
    frozen = np.asarray(frozen, dtype=bool)

    if N == 1:
        llr_val = _p1_to_llr(y[0])
        if frozen[0]:
            return [(np.array([0], dtype=int), np.array([y[0]], dtype=np.float64), 0.0)]
        paths = []
        for bit in (0, 1):
            paths.append(
                (
                    np.array([bit], dtype=int),
                    np.array([float(bit)], dtype=np.float64),
                    _pm_penalty(llr_val, bit),
                )
            )
        paths.sort(key=lambda item: (item[2], item[0][0]))
        return paths[:list_size]

    y_top = _cnop_p1(y[::2], y[1::2])
    left_paths = _scl_paths_p1(y_top, frozen[: N // 2], list_size)

    all_paths = []
    for u_left, x_left, pm_left in left_paths:
        y_bot = _vnop_p1(_cnop_p1(x_left, y[::2]), y[1::2])
        right_paths = _scl_paths_p1(y_bot, frozen[N // 2 :], list_size)
        for u_right, x_right, pm_right in right_paths:
            u = np.concatenate([u_left, u_right])
            x1 = _cnop_p1(x_left, x_right)
            x = np.zeros(N, dtype=np.float64)
            x[::2] = x1
            x[1::2] = x_right
            all_paths.append((u, x, pm_left + pm_right))

    all_paths.sort(key=lambda item: (item[2], np.sum(item[0])))
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器（P1 域递归列表译码）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        p1 = _llr_to_p1(llr_ch)
        paths = _scl_paths_p1(p1, self.frozen_bits, self.list_size)

        if self.crc_length > 0:
            valid = [(u, pm) for u, _, pm in paths if crc_check(u, self.crc_length)]
            if valid:
                u_hat, pm = min(valid, key=lambda x: x[1])
                return u_hat.copy(), pm

        u_hat, _, pm = paths[0]
        return u_hat.copy(), pm
