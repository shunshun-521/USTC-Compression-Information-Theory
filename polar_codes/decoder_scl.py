"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, _hard_decision, sc_decode
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _scl_decode_stage(self, llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            llr = llr_node[0]
            if frozen_node[0]:
                return [(np.array([0], dtype=int), np.array([0], dtype=int), 0.0)]
            return [
                (np.array([0], dtype=int), np.array([0], dtype=int), self._pm_penalty(llr, 0)),
                (np.array([1], dtype=int), np.array([1], dtype=int), self._pm_penalty(llr, 1)),
            ]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        left_paths = self._scl_decode_stage(llr_left, frozen_node[:half])

        results = []
        for u1, u1_up, pm1 in left_paths:
            llr_right = g_operation(llr_node[:half], llr_node[half:], u1_up)
            right_paths = self._scl_decode_stage(llr_right, frozen_node[half:])
            for u2, u2_up, pm2 in right_paths:
                u_hat = np.concatenate([u1, u2])
                u_up = np.concatenate([u1_up ^ u2_up, u2_up])
                results.append((u_hat, u_up, pm1 + pm2))

        results.sort(key=lambda x: x[2])
        return results[: self.list_size]

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.N > 4:
            llr_ch = llr_ch[bit_reversal_permutation(self.N)]

        stage_results = self._scl_decode_stage(llr_ch, self.frozen_bits)
        candidates = [(u, pm) for u, _, pm in stage_results]

        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, pm in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                candidates = valid

        candidates.sort(key=lambda x: x[1])
        return candidates[0]
