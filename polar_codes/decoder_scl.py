"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _reorder_llr, f_operation, g_operation, sc_decode

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


# ==================== SCL 译码器 ====================


def _active_llr_level(phi, n):
    level = 0
    p = phi
    while p % 2 == 1 and level < n:
        level += 1
        p >>= 1
    return level


class _PathState:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr_ch


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _pm_penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _update_llrs(self, path, phi):
        n = self.n
        N = self.N
        for s in range(_active_llr_level(phi, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(phi, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = float(f_operation(path.L[j, s], path.L[j + branch_size, s]))
                else:
                    path.L[j, s + 1] = float(
                        g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )
                    )

    def _update_bits(self, path, phi):
        n = self.n
        for s in range(n - 1, -1, -1):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            if phi % block_size >= branch_size:
                continue
            j = phi - phi % block_size
            path.B[j, s] = path.B[j, s + 1] ^ path.B[j + branch_size, s + 1]
            path.B[j + branch_size, s] = path.B[j + branch_size, s + 1]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = _reorder_llr(llr_ch)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llrs(path, phi)
                llr = float(path.L[phi, self.n])
                if self.frozen_bits[phi]:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), path, 0))
                else:
                    for u in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr, u), path, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, u in candidates[: self.list_size]:
                child = _PathState(self.N, self.n, llr_ch)
                child.pm = pm
                child.u_hat = parent.u_hat.copy()
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat[phi] = u
                child.B[phi, self.n] = u
                self._update_bits(child, phi)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p.u_hat[info_positions], self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat, best.pm
