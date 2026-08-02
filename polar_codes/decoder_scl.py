"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly_map = {8: 0x07, 16: 0x8005}
    if crc_length not in poly_map:
        raise ValueError("crc_length must be 8 or 16")
    poly = poly_map[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, bit_idx):
        for s in range(self.n - _active_llr_level(bit_idx, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        top_bit,
                    )

    def _update_bits(self, B, bit_idx):
        if bit_idx < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(bit_idx, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(bit_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        int(B[j, s]) ^ int(B[j - branch_size, s])
                    )
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        u_dec = 0 if llr >= 0 else 1
        return 0.0 if u == u_dec else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        paths = [{
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for bit_idx in decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path['L'], path['B'], bit_idx)
                llr = path['L'][bit_idx, self.n]

                if bit_idx in self.frozen_set:
                    path['u_hat'][bit_idx] = 0
                    path['B'][bit_idx, self.n] = 0
                    path['pm'] += self._pm_penalty(llr, 0)
                    self._update_bits(path['B'], bit_idx)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr, u_bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        child['u_hat'][bit_idx] = u_bit
                        child['B'][bit_idx, self.n] = u_bit
                        self._update_bits(child['B'], bit_idx)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            crc_ok = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            if crc_ok:
                best = crc_ok[0]

        return best['u_hat'].copy(), best['pm']
