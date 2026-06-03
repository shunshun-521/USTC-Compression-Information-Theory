"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    bit_reversal_index,
    f_operation,
    g_operation,
    sc_decode,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                reg = ((reg << 1) ^ poly) & 0xFF if (reg & 0x80) else (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                reg = ((reg << 1) ^ poly) & 0xFFFF if (reg & 0x8000) else (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_add(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm if u == u_hard else pm + abs(llr)


def _path_llr(path, phi, n, N):
    """更新路径 LLR 树并返回当前比特 LLR。"""
    L, B = path["L"], path["B"]
    l = phi
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )
    return L[l, n]


def _path_update_bits(path, phi, n, N):
    B = path["B"]
    if phi < N // 2:
        return
    for s in range(n, n - _active_bit_level(phi, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(phi, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（N×(n+1) 的 L/B 数组，与 SC 相同的因子图更新）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.phases = [bit_reversal_index(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        n, N = self.n, self.N
        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "u": np.zeros(N, dtype=np.int8), "L": L, "B": B}

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in self.phases:
            new_paths = []
            for path in paths:
                llr_bit = _path_llr(path, phi, self.n, self.N)
                if self.frozen_bits[phi]:
                    bit = 0
                    pm = _pm_add(path["pm"], llr_bit, bit)
                    cp = {
                        "pm": pm,
                        "u": path["u"].copy(),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    cp["u"][phi] = bit
                    cp["B"][phi, self.n] = bit
                    _path_update_bits(cp, phi, self.n, self.N)
                    new_paths.append(cp)
                else:
                    for bit in (0, 1):
                        pm = _pm_add(path["pm"], llr_bit, bit)
                        cp = {
                            "pm": pm,
                            "u": path["u"].copy(),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        cp["u"][phi] = bit
                        cp["B"][phi, self.n] = bit
                        _path_update_bits(cp, phi, self.n, self.N)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            paths = valid if valid else paths
        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
