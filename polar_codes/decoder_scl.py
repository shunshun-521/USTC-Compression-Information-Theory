"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ bit
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if feedback:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

        self.info_indices = np.where(~self.frozen_bits)[0]
        if crc_length > 0:
            self.crc_info_indices = self.info_indices[:-crc_length]
        else:
            self.crc_info_indices = self.info_indices

    def _path_metric_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        llr_perm = np.empty(N, dtype=np.float64)
        for i in range(N):
            llr_perm[i] = llr_ch[_bit_reversed(i, n)]

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'u_hat': np.zeros(N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr_perm

        for i in range(N):
            l = _bit_reversed(i, n)
            new_candidates = []

            for path in paths:
                L, B = path['L'], path['B']
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                llr_bit = L[l, n]
                if self.frozen_bits[l]:
                    bit = 0
                    pm = self._path_metric_update(path['pm'], llr_bit, 0)
                    B[l, n] = 0
                    path['u_hat'][l] = 0
                    path['pm'] = pm
                    self._propagate_bits(B, l, n)
                    new_candidates.append(path)
                else:
                    for bit in (0, 1):
                        L_copy = L.copy()
                        B_copy = B.copy()
                        u_copy = path['u_hat'].copy()
                        pm = self._path_metric_update(path['pm'], llr_bit, bit)
                        B_copy[l, n] = bit
                        u_copy[l] = bit
                        self._propagate_bits(B_copy, l, n)
                        new_candidates.append({
                            'pm': pm,
                            'L': L_copy,
                            'B': B_copy,
                            'u_hat': u_copy,
                        })

            new_candidates.sort(key=lambda p: p['pm'])
            paths = new_candidates[:self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path['u_hat'][self.crc_info_indices]
                crc_valid.append(crc_check(info_bits, self.crc_length))
            else:
                crc_valid.append(True)

        if any(crc_valid):
            best = min(
                [p for p, ok in zip(paths, crc_valid) if ok],
                key=lambda p: p['pm'],
            )
        else:
            best = paths[0]

        return best['u_hat'], best['pm']

    def _propagate_bits(self, B, l, n):
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
