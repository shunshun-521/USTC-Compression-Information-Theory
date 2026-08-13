"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_llr(self, L, C, phi):
        """计算当前路径在比特 phi 处的 LLR"""
        l = _bit_reversed(phi, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], C[j - branch_size, s + 1]
                    )
        return L[l, self.n]

    def _path_update_bits(self, C, phi, bit):
        """比特回传"""
        l = _bit_reversed(phi, self.n)
        C[l, self.n] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        C[j - branch_size, s - 1] = C[j, s] ^ C[j - branch_size, s]
                        C[j, s - 1] = C[j, s]

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [{
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'C': np.zeros((N, n + 1), dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                L = path['L']
                C = path['C']
                llr_val = self._path_llr(L, C, phi)

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_path = {
                        'pm': path['pm'] + penalty,
                        'u_hat': path['u_hat'].copy(),
                        'L': L.copy(),
                        'C': C.copy(),
                    }
                    new_path['u_hat'][l] = 0
                    self._path_update_bits(new_path['C'], phi, 0)
                    new_paths.append(new_path)
                else:
                    for bit in (0, 1):
                        expected = 0 if llr_val >= 0 else 1
                        penalty = 0.0 if bit == expected else abs(llr_val)
                        new_path = {
                            'pm': path['pm'] + penalty,
                            'u_hat': path['u_hat'].copy(),
                            'L': L.copy(),
                            'C': C.copy(),
                        }
                        new_path['u_hat'][l] = bit
                        self._path_update_bits(new_path['C'], phi, bit)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            info_bits = best['u_hat'][info_positions]
            valid = [p for p in paths if crc_check(
                p['u_hat'][info_positions], self.crc_length
            )]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
