"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode, _vector_f, _vector_g, _xor_merge


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    bits = list(np.asarray(info_bits, dtype=int))
    for _ in range(crc_length):
        bits.append(0)
    for i in range(len(info_bits)):
        if bits[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    bits[i + j] ^= 1
    return np.array(bits[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ('pm', 'node_values')

    def __init__(self, N):
        self.pm = 0.0
        self.node_values = [0] * N


class SCLDecoder:
    """SCL 译码器（多路径 SC 树扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _decode_paths(self, y, depth, node, paths):
        if depth == self.n - 1:
            new_paths = []
            for path in paths:
                llr = y[0]
                if node in self.frozen_set:
                    child = _Path(self.N)
                    child.node_values = path.node_values.copy()
                    child.pm = path.pm + self._pm_penalty(llr, 0)
                    child.node_values[node] = 0
                    new_paths.append((child, [0]))
                else:
                    for bit in (0, 1):
                        child = _Path(self.N)
                        child.node_values = path.node_values.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.node_values[node] = bit
                        new_paths.append((child, [bit]))
            new_paths.sort(key=lambda item: item[0].pm)
            return new_paths[:self.list_size]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_paths = self._decode_paths(_vector_f(L1, L2), depth + 1, 2 * node, paths)

        combined = []
        for path, arr1 in left_paths:
            right_paths = self._decode_paths(_vector_g(L1, L2, arr1), depth + 1, 2 * node + 1, [path])
            for rpath, arr2 in right_paths:
                combined.append((rpath, _xor_merge(arr1, arr2)))

        combined.sort(key=lambda item: item[0].pm)
        return combined[:self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = llr_ch[self.br]
        results = self._decode_paths(list(llr), 0, 0, [_Path(self.N)])

        if self.crc_length > 0 and self.info_indices is not None:
            valid = []
            for path, _ in results:
                payload = np.array(path.node_values, dtype=int)[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p.pm) if valid else min(results, key=lambda x: x[0].pm)[0]
        elif self.crc_length > 0:
            valid = [p for p, _ in results if crc_check(np.array(p.node_values), self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(results, key=lambda x: x[0].pm)[0]
        else:
            best = min(results, key=lambda x: x[0].pm)[0]

        return np.array(best.node_values, dtype=int), best.pm
