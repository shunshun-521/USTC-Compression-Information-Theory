"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded[-crc_length:], bits[-crc_length:])


def precompute_sc_indices(N):
    """兼容接口"""
    n = int(math.log2(N))
    return [2 ** l for l in range(n + 1)], [], []


def _path_metric_penalty(llr_val, u):
    preferred = 0 if llr_val >= 0 else 1
    return 0.0 if u == preferred else abs(llr_val)


class _PathState:
    __slots__ = ("y", "f_mask", "offset", "length", "u_partial", "pm", "stage", "u_left")

    def __init__(self, y, f_mask, offset, length, u_partial, pm):
        self.y = y
        self.f_mask = f_mask
        self.offset = offset
        self.length = length
        self.u_partial = u_partial
        self.pm = pm
        self.stage = 0
        self.u_left = None


class SCLDecoder:
    """SCL 译码器：在递归树叶子处进行路径分裂"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _scl_decode_tree(self, y, f_mask, offset):
        n = len(y)
        if n == 1:
            idx = offset
            if self.frozen_bits[idx]:
                return [(np.array([0]), 0.0)]
            llr = y[0]
            paths = [
                (np.array([0]), _path_metric_penalty(llr, 0)),
                (np.array([1]), _path_metric_penalty(llr, 1)),
            ]
            paths.sort(key=lambda x: x[1])
            return paths[: self.list_size]

        u1est = f_operation(y[0::2], y[1::2])
        half = n // 2
        left_paths = self._scl_decode_tree(u1est, f_mask[:half], offset)

        all_paths = []
        for u_left, pm_left in left_paths:
            u1hp = u_left.astype(np.float64)
            u2est = g_operation(
                f_operation(u1hp, y[0::2]), y[1::2], u_left
            )
            right_paths = self._scl_decode_tree(
                u2est, f_mask[half:], offset + half
            )
            for u_right, pm_right in right_paths:
                u = np.zeros(n, dtype=int)
                u[:half] = u_left
                u[half:] = u_right
                all_paths.append((u, pm_left + pm_right))

        all_paths.sort(key=lambda x: x[1])
        return all_paths[: self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_decode_tree(
            llr_ch, self.frozen_bits, 0
        )

        if self.crc_length > 0:
            valid = [(u, pm) for u, pm in paths if crc_check(u, self.crc_length)]
            best_u, best_pm = (
                min(valid, key=lambda x: x[1]) if valid else paths[0]
            )
        else:
            best_u, best_pm = paths[0]

        return best_u.copy(), best_pm
