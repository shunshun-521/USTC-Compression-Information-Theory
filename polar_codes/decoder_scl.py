"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np

from decoder_sc import _polar_decode_sc, f_operation, g_operation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(8):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8) & 1
    if crc_length == 8:
        poly, degree = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, degree)
    crc_bits = np.array([(rem >> i) & 1 for i in range(degree - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8) & 1
    if crc_length == 8:
        poly, degree = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, degree) == 0


class _Path:
    __slots__ = ("pm", "u")

    def __init__(self, N):
        self.pm = 0.0
        self.u = np.zeros(N, dtype=np.int8)


def _path_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return abs(llr) if bit != hard else 0.0


def _scl_decode_block(llr, frozen_ind, list_size, N, offset=0):
    """对子块递归 SCL，返回路径列表（u 为长度 N 的完整向量）。"""
    n = len(frozen_ind)
    if n == 1:
        idx = offset
        llr0 = float(llr[0])
        out = []
        if frozen_ind[0]:
            p = _Path(N)
            p.pm = _path_penalty(llr0, 0)
            p.u[idx] = 0
            out.append(p)
        else:
            for bit in (0, 1):
                p = _Path(N)
                p.pm = _path_penalty(llr0, bit)
                p.u[idx] = bit
                out.append(p)
        return out

    half = n // 2
    llr1 = llr[:half]
    llr2 = llr[half:]
    f1 = frozen_ind[:half]
    f2 = frozen_ind[half:]

    upper_llr = f_operation(llr1, llr2)
    upper_paths = _scl_decode_block(upper_llr, f1, list_size, N, offset)

    all_paths = []
    for up in upper_paths:
        u_up = up.u[offset : offset + half].astype(np.float64)
        low_llr = g_operation(llr1, llr2, u_up)
        low_paths = _scl_decode_block(low_llr, f2, list_size, N, offset + half)
        for lp in low_paths:
            merged = _Path(N)
            merged.pm = up.pm + lp.pm
            merged.u = up.u.copy()
            merged.u[offset + half : offset + n] = lp.u[offset + half : offset + n]
            all_paths.append(merged)

    all_paths.sort(key=lambda p: p.pm)
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_ind = np.zeros(N, dtype=np.int8)
        self.frozen_ind[self.frozen_bits] = 1

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = _scl_decode_block(llr_ch, self.frozen_ind, self.list_size, self.N, 0)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u, self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return best.u.astype(int), best.pm
