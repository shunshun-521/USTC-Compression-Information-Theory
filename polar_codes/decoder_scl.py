"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_register(bits, crc_length=8, append_zeros=0):
    poly = CRC_POLYS[crc_length]
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        msb = reg & top
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    for _ in range(append_zeros):
        msb = reg & top
        reg = (reg << 1) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_register(info_bits, crc_length, append_zeros=crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    return _crc_register(bits, crc_length, append_zeros=0) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _copy_state(self, pm, u_hat, L, B):
        return pm, u_hat.copy(), L.copy(), B.copy()

    def decode(self, llr_ch):
        """返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch

        paths = [(0.0, np.zeros(N, dtype=int), L0, B0)]

        for l in self.decode_order:
            new_paths = []
            for pm, u_hat, L, B in paths:
                _update_llrs(L, B, l, n, N)
                llr_bit = L[l, n]

                if l in self.frozen_set:
                    u_hat[l] = 0
                    B[l, n] = 0
                    if llr_bit < 0:
                        pm += abs(llr_bit)
                    _update_bits(B, l, n, N)
                    new_paths.append((pm, u_hat, L, B))
                else:
                    preferred = 0 if llr_bit >= 0 else 1
                    for bit in (0, 1):
                        pm2, uh2, L2, B2 = self._copy_state(pm, u_hat, L, B)
                        uh2[l] = bit
                        B2[l, n] = bit
                        if bit != preferred:
                            pm2 += abs(llr_bit)
                        _update_bits(B2, l, n, N)
                        new_paths.append((pm2, uh2, L2, B2))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, u_hat, _, _ in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        pm, u_hat, _, _ = paths[0]
        return u_hat, pm
