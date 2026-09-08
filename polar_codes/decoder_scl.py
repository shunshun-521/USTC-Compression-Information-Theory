"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import LLR_MAX, f_operation, g_operation, sc_decode


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _pm_penalty(llr_val, u_bit):
    if llr_val >= 0:
        hard = 0
    elif llr_val < 0:
        hard = 1
    else:
        hard = 1
    return 0.0 if u_bit == hard else abs(llr_val)


def _compute_bit_llr(llr, frozen_ind, u_prefix, phi):
    """在给定前缀比特下计算第 phi 位的 LLR"""
    llr = np.clip(np.asarray(llr, dtype=np.float64), -LLR_MAX, LLR_MAX)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    u_work = u_prefix.copy()

    def recurse(llr_node, frozen_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if idx == phi:
                return float(llr_node[0])
            if idx < phi:
                return None
            if frozen_node[0] == 1:
                u_work[idx] = 0
            elif llr_node[0] >= 0:
                u_work[idx] = 0
            elif llr_node[0] < 0:
                u_work[idx] = 1
            else:
                u_work[idx] = 1
            return None

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        llr_left = f_operation(llr1, llr2)

        res = recurse(llr_left, frozen_node[:half], offset)
        if res is not None:
            return res

        u_left = u_work[offset : offset + half].copy()
        llr_right = g_operation(llr1, llr2, u_left)
        return recurse(llr_right, frozen_node[half:], offset + half)

    result = recurse(llr, frozen_ind, 0)
    return 0.0 if result is None else result


class Path:
    __slots__ = ("pm", "u_hat", "llr")

    def __init__(self, N, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.llr = np.clip(llr_ch, -LLR_MAX, LLR_MAX).copy()

    def copy(self):
        p = Path(len(self.u_hat), self.llr)
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = self.frozen_bits.astype(np.float64)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -LLR_MAX, LLR_MAX)
        paths = [Path(self.N, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_val = _compute_bit_llr(path.llr, self.frozen_ind, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    new_path = path.copy()
                    new_path.pm += _pm_penalty(llr_val, 0)
                    new_path.u_hat[phi] = 0
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += _pm_penalty(llr_val, u_bit)
                        new_path.u_hat[phi] = u_bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = (
                min(valid, key=lambda p: p.pm)
                if valid
                else min(paths, key=lambda p: p.pm)
            )
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
