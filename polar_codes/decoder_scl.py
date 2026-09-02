"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _SCDState,
    _update_bits,
    _update_llrs,
    bit_reversed_index,
    permute_llr_for_decode,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


def _pm_penalty(llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    return 0.0 if u == u_from_llr else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径状态浅拷贝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = permute_llr_for_decode(llr_ch)
        paths = [{"pm": 0.0, "state": self._new_state(llr_ch)}]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path["state"], l)
                llr = path["state"].L[l, self.n]

                if l in self.frozen_set:
                    p = self._branch(path, l, 0, llr, force_u=0)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        new_paths.append(self._branch(path, l, u, llr))

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["state"].B[:, self.n].astype(int)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["state"].B[:, self.n].astype(int)
                if crc_check(bits[self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["state"].B[:, self.n].astype(int)

        return u_hat, best["pm"]

    def _new_state(self, llr_ch):
        state = _SCDState(self.N)
        state.L[:, 0] = llr_ch
        return state

    def _branch(self, path, l, u, llr, force_u=None):
        u = force_u if force_u is not None else u
        new_state = copy.deepcopy(path["state"])
        pm = path["pm"] + _pm_penalty(llr, u)
        new_state.B[l, self.n] = u
        _update_bits(new_state, l)
        return {"pm": pm, "state": new_state}
