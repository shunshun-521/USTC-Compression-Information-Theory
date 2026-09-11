"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_compute(info_bits, poly, crc_length):
    """标准 MSB-first CRC 计算"""
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def _crc_to_bits(reg, crc_length):
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_compute(info_bits, poly, crc_length)
    return np.concatenate([info_bits, _crc_to_bits(reg, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_compute(bits[:-crc_length], poly, crc_length)
    received = 0
    for i, b in enumerate(bits[-crc_length:]):
        received |= int(b) << (crc_length - 1 - i)
    return expected == received


class PathState:
    __slots__ = ("pm", "node_values")

    def __init__(self, N):
        self.pm = 0.0
        self.node_values = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )

    def _update_pm(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def _decode_paths(self, y, depth, node, paths):
        if depth == self.n - 1:
            new_paths = []
            for path in paths:
                llr = float(y[0])
                if node in self.frozen_set:
                    p = PathState(self.N)
                    p.pm = self._update_pm(path.pm, llr, 0)
                    p.node_values = path.node_values.copy()
                    p.node_values[node] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = PathState(self.N)
                        p.pm = self._update_pm(path.pm, llr, bit)
                        p.node_values = path.node_values.copy()
                        p.node_values[node] = bit
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            return new_paths[: self.list_size]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)

        left_paths = self._decode_paths(left_llr, depth + 1, 2 * node, paths)

        merged = []
        for lp in left_paths:
            left_bits = self._subtree_bits(lp.node_values, 2 * node, depth + 1)
            right_llr = g_operation(L1, L2, left_bits)
            right_paths = self._decode_paths(
                right_llr, depth + 1, 2 * node + 1, [lp]
            )
            merged.extend(right_paths)

        merged.sort(key=lambda p: p.pm)
        return merged[: self.list_size]

    def _subtree_bits(self, values, node, depth):
        """返回子树已译比特向量（与 g 运算所需顺序一致）"""
        if depth == self.n - 1:
            return [values[node]]
        half = 2 ** (self.n - 1 - depth)
        left = self._subtree_bits(values, 2 * node, depth + 1)
        right = self._subtree_bits(values, 2 * node + 1, depth + 1)
        return [(left[i] + right[i]) % 2 for i in range(len(left))] + list(right)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        paths = self._decode_paths(
            np.asarray(llr_ch, dtype=np.float64), 0, 0, [PathState(self.N)]
        )

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.node_values[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.node_values, best.pm
