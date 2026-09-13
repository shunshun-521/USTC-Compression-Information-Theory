"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=int)])

    if crc_length == 8:
        reg = 0
        for bit in payload:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg == 0

    reg = 0
    for bit in payload:
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg == 0


def _pm_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    return pm if bit == hard else pm + abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

        from decoder_sc import (
            _active_bit_level,
            _active_llr_level,
            _bit_reversed,
            _lower_llr,
            _upper_llr,
        )
        self._active_bit_level = _active_bit_level
        self._active_llr_level = _active_llr_level
        self._bit_reversed = _bit_reversed
        self._lower_llr = _lower_llr
        self._upper_llr = _upper_llr

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi_nat in range(N):
            l = self._bit_reversed(phi_nat, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    pm = _pm_update(path['pm'], llr, 0)
                    p = self._copy_path(path)
                    p['pm'] = pm
                    p['B'][l, n] = 0
                    p['u'][l] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        pm = _pm_update(path['pm'], llr, bit)
                        p = self._copy_path(path)
                        p['pm'] = pm
                        p['B'][l, n] = bit
                        p['u'][l] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u'][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u'], best['pm']

    def _copy_path(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u': path['u'].copy(),
        }

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - self._active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path['L'][j, s + 1] = self._upper_llr(
                        path['L'][j, s], path['L'][j + branch_size, s]
                    )
                else:
                    path['L'][j, s + 1] = self._lower_llr(
                        path['L'][j, s],
                        path['L'][j - branch_size, s],
                        int(path['B'][j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - self._active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path['B'][j - branch_size, s - 1] = (
                        int(path['B'][j, s]) ^ int(path['B'][j - branch_size, s])
                    )
                    path['B'][j, s - 1] = path['B'][j, s]
