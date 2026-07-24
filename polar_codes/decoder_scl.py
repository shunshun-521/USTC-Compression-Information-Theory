"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import sc_decode, _permute_llr_from_channel, f_boxplus, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bit(crc, bit, poly, width):
    msb = (crc >> (width - 1)) & 1
    crc = ((crc << 1) | int(bit)) & ((1 << width) - 1)
    if msb:
        crc ^= poly
    return crc


def _crc_encode_bits(info_bits, width, poly):
    crc = 0
    for bit in info_bits:
        crc = _crc_bit(crc, bit, poly, width)
    for _ in range(width):
        crc = _crc_bit(crc, 0, poly, width)
    return np.array([(crc >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)


def _crc_check_bits(bits, width, poly):
    crc = 0
    for bit in bits:
        crc = _crc_bit(crc, bit, poly, width)
    return crc == 0


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        crc_bits = _crc_encode_bits(info_bits, 8, _CRC8_POLY)
    else:
        crc_bits = _crc_encode_bits(info_bits, 16, _CRC16_POLY)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return _crc_check_bits(bits, 8, _CRC8_POLY)
    return _crc_check_bits(bits, 16, _CRC16_POLY)


class _PathState:
    __slots__ = ("pm", "u_hat", "llr_cache", "bits_cache")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.llr_cache = {}
        self.bits_cache = {}


def _pm_update(pm, llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    if bit != hard:
        pm += abs(llr_val)
    return pm


def _decode_bit_llr(llr_ch, frozen_bits, u_hat, phi):
    """计算第 phi 位的 LLR（基于已译码前缀）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(math.log2(len(llr_ch)))

    def calc(layer, idx, depth_bits):
        if layer == n:
            return llr_ch[idx]
        if ((depth_bits >> layer) & 1) == 0:
            return f_boxplus(
                calc(layer + 1, 2 * idx, depth_bits),
                calc(layer + 1, 2 * idx + 1, depth_bits),
            )
        u_val = u_hat[idx * (1 << (n - layer - 1))]
        return g_operation(
            calc(layer + 1, 2 * idx, depth_bits),
            calc(layer + 1, 2 * idx + 1, depth_bits),
            u_val,
        )

    return calc(0, 0, phi)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = _permute_llr_from_channel(llr_ch)
        paths = [_PathState(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_val = _decode_bit_llr(llr_ch, self.frozen_bits, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    p = _PathState(self.N)
                    p.u_hat = path.u_hat.copy()
                    p.pm = _pm_update(path.pm, llr_val, 0)
                    p.u_hat[phi] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = _PathState(self.N)
                        p.u_hat = path.u_hat.copy()
                        p.pm = _pm_update(path.pm, llr_val, bit)
                        p.u_hat[phi] = bit
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
