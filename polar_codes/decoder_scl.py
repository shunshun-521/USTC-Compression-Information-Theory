"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    align_llr_for_decoder,
    active_llr_level,
    active_bit_level,
    bit_reversed_index,
)


_CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC_POLYS.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
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
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, u_bit):
    """路径度量更新：与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = align_llr_for_decoder(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        L = self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed_index(phi, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)

                llr_leaf = path['L'][l, n]
                if self.frozen_bits[l]:
                    pm = _pm_update(path['pm'], llr_leaf, 0)
                    child = self._lazy_copy(path)
                    child['pm'] = pm
                    child['B'][l, n] = 0
                    child['u'][l] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pm = _pm_update(path['pm'], llr_leaf, u_bit)
                        child = self._lazy_copy(path)
                        child['pm'] = pm
                        child['B'][l, n] = u_bit
                        child['u'][l] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]

        best = self._select_best_path(paths)
        return best['u'], best['pm']

    def _lazy_copy(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u': path['u'].copy(),
        }

    def _update_llrs(self, path, l):
        L_arr = path['L']
        B_arr = path['B']
        n = self.n
        N = self.N

        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L_arr[j, s + 1] = f_operation(L_arr[j, s], L_arr[j + branch_size, s])
                else:
                    top_bit = B_arr[j - branch_size, s + 1]
                    L_arr[j, s + 1] = g_operation(
                        L_arr[j - branch_size, s], L_arr[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B_arr = path['B']
        n = self.n
        N = self.N
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B_arr[j - branch_size, s - 1] = int(B_arr[j, s]) ^ int(
                        B_arr[j - branch_size, s]
                    )
                    B_arr[j, s - 1] = B_arr[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            crc_ok = [
                p for p in paths
                if crc_check(p["u"][self.info_positions], self.crc_length)
            ]
            if crc_ok:
                return min(crc_ok, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
