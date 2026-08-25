"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from scd_core import (
    bit_reversal_permutation,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    bit_reversed_index,
)

# CRC 多项式
_CRC_POLY = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC_POLY.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = []
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            crc_bits.append(1)
            reg = ((reg << 1) ^ poly) & mask
        else:
            crc_bits.append(0)
            reg = (reg << 1) & mask

    crc_arr = np.array(crc_bits, dtype=np.int8)
    return np.concatenate([info_bits, crc_arr])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC_POLY.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _prepare_llr(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        return np.asarray(llr_ch, dtype=np.float64)[br]

    def _init_path(self, llr):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=np.int8)}

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(top_bit)
                    )

    def _update_bits(self, path, l):
        B = path['B']
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr_val, u):
        u_pred = 0 if llr_val >= 0 else 1
        return 0.0 if u == u_pred else abs(llr_val)

    def decode(self, llr_ch):
        llr = self._prepare_llr(llr_ch)
        paths = [self._init_path(llr)]

        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._pm_penalty(llr_val, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr_val, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, self.n] = u
                        new_path['u_hat'][l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
