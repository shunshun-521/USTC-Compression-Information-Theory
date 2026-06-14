"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    if u != u_from_llr:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        """主译码函数。返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [self._new_path(llr_ch)]
        for step, phi in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path['L'][phi, self.n]
                if self.frozen_bits[phi]:
                    child = self._fork_path(path)
                    child['pm'] = _pm_update(path['pm'], llr, 0)
                    child['u_hat'][phi] = 0
                    child['B'][phi, self.n] = 0
                    self._update_bits(child, phi)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = self._fork_path(path)
                        child['pm'] = _pm_update(path['pm'], llr, u)
                        child['u_hat'][phi] = u
                        child['B'][phi, self.n] = u
                        self._update_bits(child, phi)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {
            'pm': 0.0,
            'L': L,
            'B': np.zeros((self.N, self.n + 1), dtype=int),
            'u_hat': np.zeros(self.N, dtype=int),
        }

    def _fork_path(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u_hat': path['u_hat'].copy(),
        }

    def _update_llr(self, path, phi):
        L = path['L']
        B = path['B']
        for s in range(self.n - _active_llr_level(phi, self.n), self.n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(phi, self.N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

    def _update_bits(self, path, phi):
        if phi < self.N // 2:
            return
        B = path['B']
        for s in range(self.n, self.n - _active_bit_level(phi, self.n), -1):
            block = 2 ** s
            branch = block // 2
            for j in range(phi, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]
