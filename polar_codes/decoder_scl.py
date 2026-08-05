"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _update_llrs(L, B, l, n):
    """更新路径 LLR（与 SC 相同）。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    """比特回传（与 SC 相同）。"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _path_metric_penalty(llr, u_bit):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u_bit == preferred else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        N, n, L_size = self.N, self.n, self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for idx in range(N):
            l = _bit_reversed(idx, n)
            new_paths = []

            for path in paths:
                _update_llrs(path['L'], path['B'], l, n)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    pm = path['pm'] + _path_metric_penalty(llr, 0)
                    new_path = {
                        'pm': pm,
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    new_path['u_hat'][l] = 0
                    _update_bits(new_path['B'], l, n, N)
                    new_paths.append(new_path)
                else:
                    for u_bit in (0, 1):
                        pm = path['pm'] + _path_metric_penalty(llr, u_bit)
                        new_path = {
                            'pm': pm,
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, n] = u_bit
                        new_path['u_hat'][l] = u_bit
                        _update_bits(new_path['B'], l, n, N)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
