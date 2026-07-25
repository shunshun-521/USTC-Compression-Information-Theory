"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= (poly & ((1 << crc_length) - 1))

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_from_llr = 1 if llr < 0 else 0
        return 0.0 if u == u_from_llr else abs(llr)

    def _xor_combine(self, left, right):
        left = np.asarray(left, dtype=int)
        right = np.asarray(right, dtype=int)
        return np.concatenate([(left + right) % 2, right])

    def _prune(self, paths):
        paths.sort(key=lambda p: p['pm'])
        return paths[:self.list_size]

    def _scl_recursive(self, paths, y, depth, node):
        if depth == self.n - 1:
            new_paths = []
            for path in paths:
                if node in self.frozen_set:
                    bit = 0
                    nv = path['node_values'].copy()
                    nv[node] = bit
                    new_paths.append({
                        'pm': path['pm'] + self._pm_penalty(y[0], bit),
                        'node_values': nv,
                    })
                else:
                    for bit in (0, 1):
                        nv = path['node_values'].copy()
                        nv[node] = bit
                        new_paths.append({
                            'pm': path['pm'] + self._pm_penalty(y[0], bit),
                            'node_values': nv,
                        })
            pruned = self._prune(new_paths)
            return [(p, np.array([p['node_values'][node]], dtype=int)) for p in pruned]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]

        left_expanded = []
        for path in paths:
            left_expanded.extend(
                self._scl_recursive([path], f_operation(l1, l2), depth + 1, 2 * node)
            )
        left_expanded = left_expanded[:self.list_size * 2]
        left_expanded.sort(key=lambda x: x[0]['pm'])
        left_expanded = left_expanded[:self.list_size]

        right_all = []
        for path, left_bits in left_expanded:
            right_llr = g_operation(l1, l2, left_bits)
            right_results = self._scl_recursive([path], right_llr, depth + 1, 2 * node + 1)
            for rp, right_bits in right_results:
                combined = self._xor_combine(left_bits, right_bits)
                right_all.append((rp, combined))

        right_all.sort(key=lambda x: x[0]['pm'])
        return right_all[:self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init_paths = [{'pm': 0.0, 'node_values': np.zeros(self.N, dtype=int)}]
        results = self._scl_recursive(init_paths, llr_ch, 0, 0)

        paths = [r[0] for r in results]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['node_values'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['node_values'].copy(), best['pm']
