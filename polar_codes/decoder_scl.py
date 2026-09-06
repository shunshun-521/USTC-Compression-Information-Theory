"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversed, bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _upper_llr,
    _update_llrs,
    _update_bits,
    _frozen_set_from_array,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_process(bits, init=0):
    crc = init
    for bit in bits:
        feedback = (crc >> 7) & 1
        crc = (crc << 1) & 0xFF
        if int(bit) ^ feedback:
            crc ^= CRC8_POLY
    return crc


def _crc16_process(bits, init=0):
    crc = init
    for bit in bits:
        feedback = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if int(bit) ^ feedback:
            crc ^= CRC16_POLY
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")

    if crc_length == 8:
        crc = _crc8_process(info_bits)
        crc_bits = np.array([(crc >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    else:
        crc = _crc16_process(info_bits)
        crc_bits = np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        return _crc8_process(bits) == 0
    if crc_length == 16:
        return _crc16_process(bits) == 0
    raise ValueError("crc_length must be 8 or 16")


def _pm_update(pm, llr, u_bit):
    """路径度量更新：与 LLR 不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        br = bit_reversal_permutation(N)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch[br]
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, use_min_sum=False):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_set_from_array(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.use_min_sum = use_min_sum
        self.f_fn = f_operation if use_min_sum else _upper_llr
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi_nat in range(self.N):
            l = bit_reversed(phi_nat, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs_path(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    path.pm = _pm_update(path.pm, llr, u_bit)
                    path.u_hat[l] = u_bit
                    self._update_bits_path(path, l, u_bit)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, llr, u_bit)
                        child.u_hat[l] = u_bit
                        self._update_bits_path(child, l, u_bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _update_llrs_path(self, path, l):
        _update_llrs(path.L, path.B, l, self.n, self.f_fn)

    def _update_bits_path(self, path, l, u_bit):
        path.B[l, self.n] = u_bit
        _update_bits(path.B, l, self.n)
