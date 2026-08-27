"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_process(bits, crc_length):
    poly = _CRC_POLYS[crc_length]
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_process(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    return _crc_process(bits, crc_length) == 0


def _xor_combine(left_bits, right_bits):
    """SC 译码树中的 xor 合并"""
    res = [(left_bits[i] + right_bits[i]) % 2 for i in range(len(left_bits))]
    res.extend(right_bits)
    return res


class SCLDecoder:
    """SCL 译码器（基于 SC 递归结构的多路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == hard else abs(llr))

    def _decode_paths(self, y, depth, node, paths):
        if depth == self.n - 1:
            new_paths = []
            llr = y[0]
            for pm, nv, _ in paths:
                if node in self.frozen_set:
                    bit = 0
                    new_nv = dict(nv)
                    new_nv[node] = bit
                    new_paths.append((self._pm_update(pm, llr, bit), new_nv, [bit]))
                else:
                    for bit in (0, 1):
                        new_nv = dict(nv)
                        new_nv[node] = bit
                        new_paths.append((self._pm_update(pm, llr, bit), new_nv, [bit]))
            new_paths.sort(key=lambda x: x[0])
            return new_paths[: self.list_size]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(np.array(L1), np.array(L2)).tolist()
        left_paths = self._decode_paths(left_llr, depth + 1, 2 * node, paths)

        all_paths = []
        for pm, nv, left_xor in left_paths:
            right_llr = g_operation(np.array(L1), np.array(L2), left_xor).tolist()
            right_paths = self._decode_paths(right_llr, depth + 1, 2 * node + 1, [(pm, nv, [])])
            for rpm, rnv, right_xor in right_paths:
                combined = _xor_combine(left_xor, right_xor)
                all_paths.append((rpm, rnv, combined))

        all_paths.sort(key=lambda x: x[0])
        return all_paths[: self.list_size]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = self._decode_paths(llr_ch.tolist(), 0, 0, [(0.0, {}, [])])

        candidates = []
        for pm, nv, _ in paths:
            u_hat = np.zeros(self.N, dtype=int)
            for nd, bit in nv.items():
                u_hat[nd] = bit
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates
                     if crc_check(u[self.info_indices], self.crc_length)]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
