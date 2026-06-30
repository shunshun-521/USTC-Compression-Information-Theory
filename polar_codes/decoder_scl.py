"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
import crcmod

from encoder import bit_reversed
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005
_CRC8_FUNC = crcmod.predefined.mkCrcFun("crc-8")
_CRC16_FUNC = crcmod.predefined.mkCrcFun("crc-16")


def _pack_bits(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    nbytes = (len(bits) + 7) // 8
    return val.to_bytes(nbytes, "big")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    func = _CRC8_FUNC if crc_length == 8 else _CRC16_FUNC
    crc_val = func(_pack_bits(info_bits))
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int32)
    func = _CRC8_FUNC if crc_length == 8 else _CRC16_FUNC
    return func(_pack_bits(bits)) == 0


def _update_llrs_path(L, B, l, n):
    """对单条路径更新 LLR。"""
    N = L.shape[0]
    start_s = n - active_llr_level(l, n)
    for s in range(start_s, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


class PathState:
    """单条译码路径。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy(self):
        new = PathState.__new__(PathState)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        paths = [PathState(self.N, n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, n)
                llr_bit = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr_bit, 0)
                    new_path.B[l, n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, n)
                    new_paths.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr_bit, u)
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        _update_bits(new_path.B, l, n)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        return self._select_best_path(paths)

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
                return best.u_hat.copy(), best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
