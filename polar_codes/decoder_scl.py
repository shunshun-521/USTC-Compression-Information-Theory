"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import _SCDState, bit_reversed_index


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


def _pm_add(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（路径复制 + Permuted SC 状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is not None:
            br = bit_reversal_permutation(N)
            self.payload_indices = br[np.asarray(info_indices, dtype=int)]
        else:
            self.payload_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, _SCDState(llr_ch, self.frozen_bits, self.n))]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []
            for pm, state in paths:
                state._update_llrs(l)
                llr = state.L[l, self.n]
                if self.frozen_bits[l]:
                    pm2 = _pm_add(pm, llr, 0)
                    state.B[l, self.n] = 0
                    state._update_bits(l)
                    new_paths.append((pm2, state))
                else:
                    for u in (0, 1):
                        st = copy.deepcopy(state)
                        pm2 = _pm_add(pm, llr, u)
                        st.B[l, self.n] = u
                        st._update_bits(l)
                        new_paths.append((pm2, st))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x[0])
        if self.crc_length > 0:
            valid = []
            for pm, state in paths:
                u_hat = state.B[:, self.n].astype(int)
                if crc_check(u_hat[self.payload_indices], self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                return valid[0][1], valid[0][0]

        best_pm, best_state = paths[0]
        return best_state.B[:, self.n].astype(int), best_pm
