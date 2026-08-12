"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly

    crc_bits = np.zeros(crc_length, dtype=int)
    for i in range(crc_length):
        crc_bits[crc_length - 1 - i] = (reg >> i) & 1

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_path(self, llr_ch):
        """Initialize a single decoding path."""
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int32)
        L[:, 0] = llr_ch
        return {'pm': 0.0, 'L': L, 'B': B, 'u_hat': np.zeros(self.N, dtype=np.int32)}

    def _copy_path(self, path):
        """Shallow copy of a path (lazy copy for L/B)."""
        return {
            'pm': path['pm'],
            'L': path['L'],
            'B': path['B'],
            'u_hat': path['u_hat'].copy(),
        }

    def _deep_copy_path(self, path):
        """Deep copy when path will be modified."""
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u_hat': path['u_hat'].copy(),
        }

    def _update_llrs(self, path, l):
        """Update LLRs for bit index l."""
        L = path['L']
        B = path['B']
        n = self.n
        N = self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        """Propagate hard decision back through the tree."""
        B = path['B']
        n = self.n
        N = self.N

        if l < N // 2:
            return

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_update(self, pm, llr_val, u_decided):
        """Update path metric."""
        u_from_llr = 0 if llr_val >= 0 else 1
        if u_decided != u_from_llr:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path['L'][l, self.n]

                if l in self.frozen_set:
                    new_path = self._deep_copy_path(path)
                    new_path['pm'] = self._pm_update(path['pm'], llr_val, 0)
                    new_path['B'][l, self.n] = 0
                    new_path['u_hat'][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_cand in [0, 1]:
                        new_path = self._deep_copy_path(path)
                        new_path['pm'] = self._pm_update(path['pm'], llr_val, u_cand)
                        new_path['B'][l, self.n] = u_cand
                        new_path['u_hat'][l] = u_cand
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_passed = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            best = min(crc_passed if crc_passed else paths, key=lambda p: p['pm'])
        else:
            best = paths[0]

        return best['u_hat'], best['pm']
