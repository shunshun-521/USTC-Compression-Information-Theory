"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _frozen_to_decode_set, sc_decode


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


def _merge_paths(left, right):
    """合并左右子树路径（路径度量相加，比特 xor 合并）。"""
    pm = left[0] + right[0]
    bits = [(a + b) % 2 for a, b in zip(left[1], right[1])] + right[1]
    nodes = left[2].copy()
    nodes.update(right[2])
    return (pm, bits, nodes)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.depth_limit = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.br = bit_reversal_permutation(N)
        self.frozen_set = _frozen_to_decode_set(self.frozen_bits, self.br)

    def _decode_paths(self, y, depth, node):
        if depth == self.depth_limit - 1:
            paths = []
            if node in self.frozen_set:
                if y[0] >= 0:
                    paths.append((0.0, [0], {node: 0}))
                else:
                    paths.append((abs(y[0]), [0], {node: 0}))
            else:
                if y[0] < 0:
                    paths.append((0.0, [1], {node: 1}))
                    paths.append((abs(y[0]), [0], {node: 0}))
                else:
                    paths.append((0.0, [0], {node: 0}))
                    paths.append((abs(y[0]), [1], {node: 1}))
            paths.sort(key=lambda p: p[0])
            return paths[: self.list_size]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left_paths = self._decode_paths(f_operation(l1, l2), depth + 1, 2 * node)

        merged = []
        for lp in left_paths:
            right_in = g_operation(l1, l2, np.asarray(lp[1], dtype=np.int8))
            right_paths = self._decode_paths(right_in, depth + 1, 2 * node + 1)
            for rp in right_paths:
                merged.append(_merge_paths(lp, rp))

        merged.sort(key=lambda p: p[0])
        return merged[: self.list_size]

    def decode(self, llr_ch):
        """主译码函数。返回自然序 u_hat, pm"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = self._decode_paths(np.asarray(llr_ch, dtype=np.float64), 0, 0)
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        candidates = []
        for pm, _, nodes in paths:
            full = np.zeros(self.N, dtype=int)
            for idx, bit in nodes.items():
                full[idx] = bit
            u_nat = full[self.br]
            candidates.append((pm, u_nat))

        if self.crc_length > 0 and self.info_indices is not None:
            valid = [(pm, u) for pm, u in candidates if crc_check(u[self.info_indices], self.crc_length)]
            if valid:
                candidates = valid

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u, best_pm
