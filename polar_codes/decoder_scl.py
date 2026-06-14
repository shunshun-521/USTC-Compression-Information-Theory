"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    _update_llrs,
    _update_bits,
    _llr_to_bit,
    _pm_penalty,
    f_boxplus,
    g_operation,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        return {"L": L, "B": B, "pm": 0.0, "u": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        """主译码函数，返回自然顺序的 u_hat 与路径度量。"""
        paths = [self._init_path(llr_ch)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr_l = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    candidates.append((path["pm"] + _pm_penalty(llr_l, 0), path, 0))
                else:
                    for u_bit in (0, 1):
                        candidates.append((path["pm"] + _pm_penalty(llr_l, u_bit), path, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            parent_use = {}
            for new_pm, parent, u_bit in candidates:
                used = parent_use.get(id(parent), 0)
                if used > 0:
                    L_new = parent["L"].copy()
                    B_new = parent["B"].copy()
                else:
                    L_new = parent["L"]
                    B_new = parent["B"]
                parent_use[id(parent)] = used + 1

                u_new = parent["u"].copy()
                u_new[l] = u_bit
                B_new[l, self.n] = u_bit
                _update_bits(B_new, l, self.n)

                new_paths.append({"L": L_new, "B": B_new, "pm": new_pm, "u": u_new})

            paths = new_paths

        pm_list = [p["pm"] for p in paths]
        u_list = [p["u"] for p in paths]

        best_idx = int(np.argmin(pm_list))
        if self.crc_length > 0:
            valid = []
            for i, u_hat in enumerate(u_list):
                info_bits = u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(i)
            if valid:
                best_idx = min(valid, key=lambda i: pm_list[i])

        return u_list[best_idx], pm_list[best_idx]
