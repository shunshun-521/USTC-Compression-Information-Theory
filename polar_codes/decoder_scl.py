"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


# ==================== CRC 工具 ====================

_CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_poly_bits(crc_length):
    poly = _CRC_POLYS[crc_length]
    return [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]


def _crc_core(bits, crc_length):
    """LFSR CRC 计算，返回 crc_length 位余数"""
    poly_bits = _crc_poly_bits(crc_length)
    reg = [0] * crc_length
    for bit in bits:
        fb = reg[-1] ^ int(bit)
        reg = [fb] + reg[:-1]
        if fb:
            reg = [(r ^ p) & 1 for r, p in zip(reg, poly_bits)]
    return np.array(reg, dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_core(np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC（与 crc_encode 配套）"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _path_metric_update(pm, llr, u):
    """路径度量更新"""
    hard = 0 if llr >= 0 else 1
    if u == hard:
        return pm
    return pm + abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int8),
            'u': np.zeros(N, dtype=np.int8),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path['L'][l, n]
                if l in self.frozen_set:
                    pm = _path_metric_update(path['pm'], llr, 0)
                    child = self._fork_path(path)
                    child['pm'] = pm
                    child['u'][l] = 0
                    child['B'][l, n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_val in (0, 1):
                        pm = _path_metric_update(path['pm'], llr, u_val)
                        child = self._fork_path(path)
                        child['pm'] = pm
                        child['u'][l] = u_val
                        child['B'][l, n] = u_val
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p['u'][self.info_indices] if self.info_indices is not None else p['u']
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p['pm'])
            else:
                best = min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u'].astype(int), best['pm']

    def _fork_path(self, path):
        """Lazy copy：仅复制必要状态"""
        return {
            'pm': path['pm'],
            'L': path['L'],
            'B': path['B'].copy(),
            'u': path['u'].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path['L'], path['B']
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path['B']
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
