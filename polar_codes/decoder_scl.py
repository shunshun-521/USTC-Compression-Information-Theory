"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if (reg >> (crc_length - 1)) & 1:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i, bit in enumerate(bits[-crc_length:]):
        received |= int(bit) << (crc_length - 1 - i)
    return remainder == received


def _frozen_set_from_bits(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        fb = fb.astype(bool)
    return set(np.where(fb)[0])


def _merge_paths(left, right):
    merged = [(left[i] + right[i]) % 2 for i in range(len(left))]
    merged.extend(right)
    return merged


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits)
        self.F = _frozen_set_from_bits(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices) if info_indices is not None else None

    def _path_metric_update(self, pm, llr, u_val):
        hard = 1 if llr < 0 else 0
        if u_val != hard:
            pm += abs(llr)
        return pm

    def _left_merge(self, llr, depth, node, start, end, u_known):
        """构建左子树 merge 返回值（所有索引 < end 已知）"""

        def decode(y, d, n, s):
            if d == self.n - 1:
                if n in self.F:
                    return [0]
                return [u_known[n]]

            half = len(y) // 2
            L1, L2 = y[:half], y[half:]
            left_llr = f_operation(L1, L2)
            arr1 = decode(left_llr, d + 1, 2 * n, s)
            right_llr = g_operation(L1, L2, arr1)
            arr2 = decode(right_llr, d + 1, 2 * n + 1, s + half)
            return _merge_paths(arr1, arr2)

        return decode(llr, depth, node, start)

    def _llr_at_phi(self, llr, u_known, phi):
        """计算比特 phi 处的 LLR"""

        def decode(y, depth, node, start):
            if depth == self.n - 1:
                return y[0]

            half = len(y) // 2
            mid = start + half
            L1, L2 = y[:half], y[half:]
            left_llr = f_operation(L1, L2)

            if phi < mid:
                return decode(left_llr, depth + 1, 2 * node, start)

            arr1 = self._left_merge(left_llr, depth + 1, 2 * node, start, mid, u_known)
            right_llr = g_operation(L1, L2, arr1)
            return decode(right_llr, depth + 1, 2 * node + 1, mid)

        return decode(llr, 0, 0, 0)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        if L == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{'pm': 0.0, 'u': np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = self._llr_at_phi(llr_ch, path['u'], phi)

                if phi in self.F:
                    new_u = path['u'].copy()
                    new_u[phi] = 0
                    pm = self._path_metric_update(path['pm'], llr_phi, 0)
                    candidates.append({'pm': pm, 'u': new_u})
                else:
                    for u_val in (0, 1):
                        new_u = path['u'].copy()
                        new_u[phi] = u_val
                        pm = self._path_metric_update(path['pm'], llr_phi, u_val)
                        candidates.append({'pm': pm, 'u': new_u})

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L]

        if self.crc_length > 0:
            crc_ok = [p for p in paths if crc_check(p['u'], self.crc_length)]
            best = min(crc_ok if crc_ok else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u'].copy(), best['pm']
