"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
    sc_decode_core,
)

# ==================== CRC 工具 ====================

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly, reg_bits = CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (reg_bits - 1)
        for _ in range(8):
            if reg & (1 << (reg_bits - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << reg_bits) - 1)
            else:
                reg = (reg << 1) & ((1 << reg_bits) - 1)

    crc_bits = np.array(
        [(reg >> (reg_bits - 1 - i)) & 1 for i in range(reg_bits)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly, reg_bits = CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in bits:
        reg ^= int(bit) << (reg_bits - 1)
        for _ in range(8):
            if reg & (1 << (reg_bits - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << reg_bits) - 1)
            else:
                reg = (reg << 1) & ((1 << reg_bits) - 1)
    return reg == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """
    SCL 译码器（路径级复制 L/B 状态，与 SC 相同的因子图更新）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _path_metric(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _new_path(self, llr_ch):
        n, N = self.n, self.N
        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=np.int8)
        L[:, 0] = llr_ch[self.br]
        return {"L": L, "B": B, "pm": 0.0}

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 最优路径估计
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, Lmax = self.n, self.N, self.list_size

        paths = [self._new_path(llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                pm = path["pm"]

                _update_llrs(L, B, l, n, N)
                cur_llr = L[l, n]

                if self.frozen_bits[l]:
                    candidates = [0]
                else:
                    candidates = [0, 1]

                for bit in candidates:
                    Lc = L.copy()
                    Bc = B.copy()
                    Bc[l, n] = bit
                    Lc[l, n] = cur_llr  # 保持 LLR 不变
                    pm_new = self._path_metric(pm, cur_llr, bit)
                    _update_bits(Bc, l, n, N)
                    new_paths.append({"L": Lc, "B": Bc, "pm": pm_new})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:Lmax]
            if not paths:
                paths = new_paths[:1]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = p["B"][:, n].astype(int)
                bits = u[self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]
