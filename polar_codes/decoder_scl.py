"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import cn_boxplus, g_operation, sc_decode_recursive, LLR_MAX

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_bits(data_bits):
    crc = 0
    for b in np.asarray(data_bits, dtype=int):
        crc ^= int(b) << 7
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _crc16_bits(data_bits):
    crc = 0
    for b in np.asarray(data_bits, dtype=int):
        crc ^= int(b) << 15
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc8_bits(info_bits)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc16_bits(info_bits)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return _crc8_bits(bits) == 0
    if crc_length == 16:
        return _crc16_bits(bits) == 0
    raise ValueError("crc_length must be 8 or 16")


def _leaf_llr(llr, u_hat, phi, n, N):
    """计算第 phi 位的 SC 叶子 LLR。"""

    def rec(node, offset, length, target):
        if length == 1:
            return float(node[0])
        h = length // 2
        if target < offset + h:
            left = cn_boxplus(node[:h], node[h:])
            return rec(left, offset, h, target)
        right = g_operation(node[:h], node[h:], u_hat[offset : offset + h])
        return rec(right, offset + h, h, target)

    return rec(np.asarray(llr, dtype=np.float64), 0, N, phi)


def scl_decode(llr_ch, frozen_bits, list_size=4, crc_length=0):
    """SCL 译码，返回 (u_hat, pm)。"""
    llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -LLR_MAX, LLR_MAX)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_idx = np.where(~frozen_bits)[0]
    L = list_size

    paths = [(0.0, np.zeros(N, dtype=int))]

    for phi in range(N):
        nxt = []
        for pm, u in paths:
            llr_bit = _leaf_llr(llr_ch, u, phi, n, N)
            if frozen_bits[phi]:
                pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                u2 = u.copy()
                u2[phi] = 0
                nxt.append((pm + pen, u2))
            else:
                for bit in (0, 1):
                    pen = 0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit)
                    u2 = u.copy()
                    u2[phi] = bit
                    nxt.append((pm + pen, u2))
        nxt.sort(key=lambda x: x[0])
        paths = nxt[:L]

    if crc_length > 0:
        K_payload = len(info_idx) - crc_length
        valid = []
        for pm, u in paths:
            payload = u[info_idx[:K_payload]]
            crc_part = u[info_idx[K_payload : K_payload + crc_length]]
            if crc_check(np.concatenate([payload, crc_part]), crc_length):
                valid.append((pm, u))
        if valid:
            paths = valid

    best = min(paths, key=lambda x: x[0])
    return best[1], best[0]


class SCLDecoder:
    """SCL 译码器（含 CRC 辅助 CA-SCL）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0
        return scl_decode(llr_ch, self.frozen_bits, self.list_size, self.crc_length)
