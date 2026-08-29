"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    hard_decision,
    sc_decode,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """逐比特 CRC 处理（MSB 优先，poly 为反射多项式）"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def _get_crc_poly(crc_length):
    # CRC-8: 0x07 的反射形式为 0xE0；CRC-16: 0x8005 的反射形式为 0xA001
    return 0xE0 if crc_length == 8 else 0xA001


def _crc_encode_bits(info_bits, crc_length):
    poly = _get_crc_poly(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = _crc_process(padded, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def _crc_remainder(bits, crc_length):
    poly = _get_crc_poly(crc_length)
    return _crc_process(bits, poly, crc_length)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    return _crc_encode_bits(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化，基于 SCD 算法框架）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "u_hat": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr, 0)
                    path["pm"] += penalty
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    self._update_bits(path["B"], l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = {
                            "pm": path["pm"],
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        penalty = self._pm_penalty(llr, u_bit)
                        p["pm"] += penalty
                        p["u_hat"][l] = u_bit
                        p["B"][l, self.n] = u_bit
                        self._update_bits(p["B"], l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
