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


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _pm_update(pm, llr, bit):
    """路径度量更新：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        if self.list_size == 1:
            from decoder_sc import sc_decode
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.zeros((N, n + 1)),
            "B": np.zeros((N, n + 1), dtype=np.int32),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = self.br[i]
            candidates = []

            for pidx, path in enumerate(paths):
                L, B, pm = path["L"], path["B"], path["pm"]
                L_work = L.copy()
                B_work = B.copy()
                _update_llrs(L_work, B_work, l, n, N)
                llr_dec = L_work[l, n]

                if self.frozen_bits[i]:
                    pm_new = _pm_update(pm, llr_dec, 0)
                    B_work[l, n] = 0
                    _update_bits(B_work, l, n, N)
                    candidates.append({
                        "L": L_work, "B": B_work, "pm": pm_new,
                    })
                else:
                    for bit in (0, 1):
                        L_c = L_work.copy()
                        B_c = B_work.copy()
                        pm_new = _pm_update(pm, llr_dec, bit)
                        B_c[l, n] = bit
                        _update_bits(B_c, l, n, N)
                        candidates.append({
                            "L": L_c, "B": B_c, "pm": pm_new,
                        })

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda x: x["pm"])
        u_hat = best["B"][:, n].astype(int)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                ib = p["B"][:, n].astype(int)[info_idx]
                if crc_check(ib, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x["pm"])
                u_hat = best["B"][:, n].astype(int)

        return u_hat, best["pm"]
