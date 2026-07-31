"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import _bit_rev_indices
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _compute_u_up(u_bits):
    """根据已译码比特计算再编码部分和（与 SC 译码器一致）。"""
    u_bits = np.asarray(u_bits, dtype=int)
    n = len(u_bits)
    if n == 1:
        return u_bits.copy()
    half = n // 2
    left_up = _compute_u_up(u_bits[:half])
    right_up = _compute_u_up(u_bits[half:])
    up_left = np.bitwise_xor(left_up.astype(int), right_up.astype(int)).astype(int)
    return np.concatenate([up_left, right_up])


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class PathState:
    __slots__ = ('pm', 'u_hat')

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = PathState(len(self.u_hat))
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.brp = _bit_rev_indices(N)

    def _scl_rec(self, paths, llr, bit_offset):
        n = len(llr)
        if n == 1:
            new_paths = []
            for path in paths:
                llr_val = llr[0]
                if self.frozen_bits[bit_offset]:
                    p = path.copy()
                    p.pm += _pm_penalty(llr_val, 0)
                    p.u_hat[bit_offset] = 0
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm += _pm_penalty(llr_val, u)
                        p.u_hat[bit_offset] = u
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            return new_paths[:self.list_size]

        half = n // 2
        llr_li, llr_ri = llr[:half], llr[half:]
        x_left = f_operation(llr_li, llr_ri)
        paths = self._scl_rec(paths, x_left, bit_offset)

        all_paths = []
        for path in paths:
            u_up = _compute_u_up(path.u_hat[bit_offset:bit_offset + half])
            x_right = g_operation(llr_li, llr_ri, u_up)
            sub = self._scl_rec([path], x_right, bit_offset + half)
            all_paths.extend(sub)

        all_paths.sort(key=lambda p: p.pm)
        return all_paths[:self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.brp]
        paths = self._scl_rec([PathState(self.N)], llr_ch, 0)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
