"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, _xor_paths


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # CRC-16-IBM


def _crc_generator(crc_length):
    return CRC8_POLY if crc_length == 8 else CRC16_POLY


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    g = _crc_generator(crc_length)
    temp = list(info_bits) + [0] * crc_length
    for i in range(len(info_bits)):
        if temp[i] == 1:
            for j in range(len(g)):
                temp[i + j] ^= g[j]
    crc_bits = np.array(temp[len(info_bits) : len(info_bits) + crc_length], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    g = _crc_generator(crc_length)
    temp = list(bits)
    for i in range(len(bits) - crc_length):
        if temp[i] == 1:
            for j in range(len(g)):
                temp[i + j] ^= g[j]
    return all(x == 0 for x in temp[len(bits) - crc_length :])


def _metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _path_return(nv, node, size):
    """计算子树路径返回值（与 SC 递归 xor 合并一致）。"""
    if size == 1:
        return [int(nv[node])]
    h = size // 2
    left = _path_return(nv, 2 * node, h)
    right = _path_return(nv, 2 * node + 1, h)
    return _xor_paths(left, right)


class SCLDecoder:
    """SCL 译码器：多路径共享子树遍历，仅在叶节点分裂。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_rec(llr_ch, 0, 0, self.N, [(0.0, np.zeros(self.N, dtype=np.int8))])
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            for pm, nv in paths:
                if crc_check(nv[self.info_indices], self.crc_length):
                    return nv.astype(int), pm

        return paths[0][1].astype(int), paths[0][0]

    def _prune(self, paths):
        paths.sort(key=lambda x: x[0])
        return paths[: self.list_size]

    def _scl_rec(self, y, depth, node, size, paths):
        if depth == self.n - 1:
            llr = y[0]
            expanded = []
            for pm, nv in paths:
                if node in self.frozen_set:
                    nv = nv.copy()
                    nv[node] = 0
                    expanded.append((pm + _metric_penalty(llr, 0), nv))
                else:
                    for bit in (0, 1):
                        nv2 = nv.copy()
                        nv2[node] = bit
                        expanded.append((pm + _metric_penalty(llr, bit), nv2))
            return self._prune(expanded)

        half = size // 2
        l1 = y[:half]
        l2 = y[half:]
        left_llr = f_operation(l1, l2)
        left_paths = self._scl_rec(left_llr, depth + 1, 2 * node, half, paths)

        groups = {}
        for pm, nv in left_paths:
            arr1 = tuple(_path_return(nv, 2 * node, half))
            groups.setdefault(arr1, []).append((pm, nv))

        right_all = []
        for arr1, group in groups.items():
            right_llr = g_operation(l1, l2, list(arr1))
            right_paths = self._scl_rec(
                right_llr, depth + 1, 2 * node + 1, half, group
            )
            right_all.extend(right_paths)

        return self._prune(right_all)
