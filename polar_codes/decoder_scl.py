"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    n = len(bits)
    nbytes = (n + 7) // 8
    out = bytearray()
    for i in range(nbytes):
        byte = 0
        for j in range(8):
            idx = i * 8 + j
            byte = (byte << 1) | (int(bits[idx]) if idx < n else 0)
        out.append(byte)
    return bytes(out)


def _crc8_bytes(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    remainder = _crc8_bytes(data) if crc_length == 8 else _crc16_bytes(data)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_bytes(data) == 0
    return _crc16_bytes(data) == 0


def _pm_update(pm, llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    if u != u_from_llr:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _compute_llr(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )
        return L[l, self.n]

    def _propagate_bit(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。L=1 且无 CRC 时退化为 SC。"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                llr = self._compute_llr(path['L'], path['B'], l)

                if l in self.frozen_indices:
                    new_pm = _pm_update(path['pm'], llr, 0)
                    child = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': new_pm,
                        'u_hat': path['u_hat'].copy(),
                    }
                    child['u_hat'][l] = 0
                    child['B'][l, self.n] = 0
                    self._propagate_bit(child['B'], l)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        child = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': _pm_update(path['pm'], llr, u_bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        child['u_hat'][l] = u_bit
                        child['B'][l, self.n] = u_bit
                        self._propagate_bit(child['B'], l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
