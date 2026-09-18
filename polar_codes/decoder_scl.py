"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_logdomain, g_logdomain, f_operation, g_operation,
    bit_reversed, _active_llr_level, _active_bit_level,
)


# ==================== CRC 工具 ====================

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(data_bits, crc_len, poly):
    reg = 0
    for bit in data_bits:
        reg ^= (int(bit) << (crc_len - 1))
        for _ in range(8 if crc_len == 8 else 16):
            if crc_len == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_bits(info_bits, crc_length, poly)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================

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

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_logdomain(L[j, s], L[j + branch_size, s])
                else:
                    tb = B[j - branch_size, s + 1]
                    if np.isnan(tb):
                        tb = 0
                    L[j, s + 1] = g_logdomain(L[j, s], L[j - branch_size, s], int(tb))

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bj = B[j, s]
                    bjm = B[j - branch_size, s]
                    if np.isnan(bj):
                        bj = 0
                    if np.isnan(bjm):
                        bjm = 0
                    B[j - branch_size, s - 1] = int(bj) ^ int(bjm)
                    B[j, s - 1] = bj

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        N = self.N
        n = self.n

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch.copy()

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr_val = path['L'][l, n]

                if l in self.frozen_set:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path = {
                        'pm': path['pm'] + penalty,
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['u_hat'][l] = 0
                    new_path['B'][l, n] = 0
                    self._update_bits(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        penalty = 0.0 if ((u_bit == 0 and llr_val >= 0) or (u_bit == 1 and llr_val < 0)) else abs(llr_val)
                        new_path = {
                            'pm': path['pm'] + penalty,
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['u_hat'][l] = u_bit
                        new_path['B'][l, n] = u_bit
                        self._update_bits(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            passed = [p for p in paths if crc_check(p['u_hat'][self.info_indices], self.crc_length)]
            best = min(passed if passed else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
