"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    sc_decode,
    f_operation,
    g_operation,
    _frozen_mask,
)

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
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


def _sc_step_update(L, B, phi, n, N):
    """对单条路径执行与 sc_decode 相同的 LLR 递推，返回当前位 LLR。"""
    if phi == 0:
        for layer in range(n - 1, -1, -1):
            span = 1 << (n - layer - 1)
            for psi in range(0, N, 2 * span):
                L[layer, psi] = f_operation(L[layer + 1, psi], L[layer + 1, psi + span])
    else:
        layer = 0
        while (phi >> layer) & 1:
            layer += 1
        for i in range(layer):
            span = 1 << (n - i - 1)
            psi = phi + 1 - 2 * span
            L[i, psi + span] = g_operation(
                L[i + 1, psi], L[i + 1, psi + span], B[i + 1, psi]
            )
        for i in range(layer, n):
            span = 1 << (n - i - 1)
            psi = (phi // (2 * span)) * 2 * span
            L[i, psi] = f_operation(L[i + 1, psi], L[i + 1, psi + span])
    return L[0, phi]


def _bit_propagate(B, phi, u_bit, n):
    B[0, phi] = u_bit
    l = 0
    while (phi + 1) % (1 << (l + 1)) == 0:
        span = 1 << l
        psi = phi + 1 - 2 * span
        B[l + 1, psi] = (B[l, psi] + B[l, psi + span]) % 2
        B[l + 1, psi + span] = B[l, psi + span]
        l += 1


class SCLDecoder:
    """SCL 译码器（路径数组 + 深拷贝分裂）。 """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = []
        L0 = np.zeros((n + 1, N), dtype=np.float64)
        B0 = np.zeros((n + 1, N), dtype=np.int8)
        L0[n, :] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u": np.zeros(N, dtype=np.int8)})

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr = _sc_step_update(path["L"], path["B"], phi, n, N)
                if self.frozen_bits[phi]:
                    cand = path.copy()
                    cand["L"] = path["L"].copy()
                    cand["B"] = path["B"].copy()
                    cand["u"] = path["u"].copy()
                    cand["pm"] = _pm_update(cand["pm"], llr, 0)
                    cand["u"][phi] = 0
                    _bit_propagate(cand["B"], phi, 0, n)
                    new_paths.append(cand)
                else:
                    for u in (0, 1):
                        cand = path.copy()
                        cand["L"] = path["L"].copy()
                        cand["B"] = path["B"].copy()
                        cand["u"] = path["u"].copy()
                        cand["pm"] = _pm_update(cand["pm"], llr, u)
                        cand["u"][phi] = u
                        _bit_propagate(cand["B"], phi, u, n)
                        new_paths.append(cand)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p["u"][self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best["u"].copy(), best["pm"]
