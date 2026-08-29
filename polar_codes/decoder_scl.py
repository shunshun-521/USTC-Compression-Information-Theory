"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr_val, u_val):
    hard = 0 if llr_val >= 0 else 1
    if u_val != hard:
        pm += abs(llr_val)
    return pm


def _xor_lists(left, right):
    merged = [(left[i] + right[i]) % 2 for i in range(len(left))]
    merged.extend(right)
    return merged


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        node_values = np.zeros(self.N, dtype=int)
        results = self._decode_list(llr_ch, 0, 0, node_values)
        candidates = []
        for pm, dec, nv in results:
            candidates.append((pm, nv.copy()))
        candidates.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            valid = []
            for pm, u in candidates:
                info_bits = u[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u))
            if valid:
                candidates = valid

        pm, u_hat = candidates[0]
        return u_hat, pm

    def _decode_list(self, y, depth, node, node_values):
        """
        返回路径列表：[(pm, partial_decision_list, node_values), ...]
        """
        if depth == self.n - 1:
            paths = []
            llr_val = float(y[0])
            if node in self.frozen_set:
                node_values[node] = 0
                paths.append((0.0, [0], node_values))
            else:
                for bit in (0, 1):
                    nv = node_values.copy()
                    nv[node] = bit
                    pm = _path_metric_update(0.0, llr_val, bit)
                    paths.append((pm, [bit], nv))
            return paths

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]

        left_all = []
        for pm_base, _, nv in [(0.0, None, node_values)]:
            left_paths = self._decode_list(f_operation(l1, l2), depth + 1, 2 * node, nv)
            left_all.extend(left_paths)

        left_all.sort(key=lambda x: x[0])
        left_all = left_all[: self.list_size]

        right_all = []
        for pm, left_dec, nv in left_all:
            right_paths = self._decode_list(
                g_operation(l1, l2, left_dec), depth + 1, 2 * node + 1, nv
            )
            for rpm, rdec, rnv in right_paths:
                merged = _xor_lists(left_dec, rdec)
                right_all.append((pm + rpm, merged, rnv))

        right_all.sort(key=lambda x: x[0])
        return right_all[: self.list_size]
