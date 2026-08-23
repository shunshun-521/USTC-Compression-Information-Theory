"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _f_list, _g_list, _xor_paths
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（列表路径在子树合并后剪枝）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _leaf_extend(self, y, node, paths):
        extended = []
        for pm, bits, nv in paths:
            if node in self.frozen_set:
                nv2 = nv.copy()
                nv2[node] = 0
                pm2 = pm + (0.0 if y[0] >= 0 else abs(y[0]))
                extended.append((pm2, [0], nv2))
            else:
                for u in (0, 1):
                    nv2 = nv.copy()
                    nv2[node] = u
                    hard = 1 if y[0] < 0 else 0
                    pm2 = pm + (0.0 if u == hard else abs(y[0]))
                    extended.append((pm2, [u], nv2))
        extended.sort(key=lambda x: x[0])
        return extended[: self.list_size]

    def _decode_tree(self, y, depth, node, paths):
        if depth == self.n - 1:
            return self._leaf_extend(y, node, paths)

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = _f_list(l1, l2)

        left_paths = self._decode_tree(left, depth + 1, 2 * node, paths)

        merged = []
        for pm, bits, nv in left_paths:
            right = _g_list(l1, l2, bits)
            right_paths = self._decode_tree(
                right, depth + 1, 2 * node + 1, [(pm, bits, nv)]
            )
            for rpm, rbits, rnv in right_paths:
                merged.append(
                    (rpm, _xor_paths(bits, rbits), rnv)
                )

        merged.sort(key=lambda x: x[0])
        return merged[: self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.rev].tolist()

        paths = self._decode_tree(
            llr_br, 0, 0, [(0.0, [], np.zeros(self.N, dtype=int))]
        )
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p[2], self.crc_length)]
            best = min(valid, key=lambda x: x[0]) if valid else min(paths, key=lambda x: x[0])
        else:
            best = min(paths, key=lambda x: x[0])

        return best[2].astype(int), best[0]
