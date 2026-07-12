"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive, _xor_paths


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


class SCLDecoder:
    """SCL 译码器，树结构与 SC 递归实现保持一致。"""

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
        paths = self._decode_node(llr_ch, 0, 0, [(0.0, np.zeros(self.N, dtype=int))])
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            for pm, nv in paths:
                if crc_check(nv[self.info_indices], self.crc_length):
                    return nv, pm

        return paths[0][1], paths[0][0]

    def _prune(self, paths):
        paths.sort(key=lambda x: x[0])
        return paths[: self.list_size]

    def _decode_node(self, y, depth, node, active_paths):
        if depth == self.n - 1:
            llr = y[0]
            expanded = []
            for pm, nv in active_paths:
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

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left_llr = f_operation(l1, l2)

        left_expanded = []
        for pm, nv in active_paths:
            for pm2, nv2, arr1 in self._decode_node_with_path(
                left_llr, depth + 1, 2 * node, pm, nv
            ):
                right_llr = g_operation(l1, l2, arr1)
                for pm3, nv3, _ in self._decode_node_with_path(
                    right_llr, depth + 1, 2 * node + 1, pm2, nv2
                ):
                    left_expanded.append((pm3, nv3))

        return self._prune(left_expanded)

    def _decode_node_with_path(self, y, depth, node, pm, nv):
        """返回 (pm, nv, path_return) 三元组。"""
        if depth == self.n - 1:
            llr = y[0]
            results = []
            if node in self.frozen_set:
                nv2 = nv.copy()
                nv2[node] = 0
                results.append((pm + _metric_penalty(llr, 0), nv2, [0]))
            else:
                for bit in (0, 1):
                    nv2 = nv.copy()
                    nv2[node] = bit
                    results.append((pm + _metric_penalty(llr, bit), nv2, [bit]))
            results.sort(key=lambda x: x[0])
            return results[: self.list_size]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left_llr = f_operation(l1, l2)

        merged = []
        left_results = self._decode_node_with_path(
            left_llr, depth + 1, 2 * node, pm, nv
        )
        for pm2, nv2, arr1 in left_results:
            right_llr = g_operation(l1, l2, arr1)
            right_results = self._decode_node_with_path(
                right_llr, depth + 1, 2 * node + 1, pm2, nv2
            )
            for pm3, nv3, arr2 in right_results:
                merged.append((pm3, nv3, _xor_paths(arr1, arr2)))

        merged.sort(key=lambda x: x[0])
        return merged[: self.list_size]
