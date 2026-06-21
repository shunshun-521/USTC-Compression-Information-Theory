"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, width = CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, width)
    crc_bits = np.array([(remainder >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


def _llr_at_phi(llr_ch, u_hat, phi, n):
    """计算第 phi 个比特的 LLR（与 SC 递归译码一致）。"""
    llr = llr_ch.copy()
    result = [0.0]

    class StopAtPhi(Exception):
        pass

    def decode_layer(layer, start, length):
        if length == 1:
            idx = start
            if idx == phi:
                result[0] = llr[idx]
                raise StopAtPhi()
            return

        half = length // 2
        offset = 1 << (n - 1 - layer)

        for i in range(half):
            a = start + i
            b = a + offset
            llr[a] = f_operation(llr[a], llr[b])

        decode_layer(layer + 1, start, half)

        for i in range(half):
            a = start + i
            b = a + offset
            llr[a] = g_operation(llr[a], llr[b], u_hat[a])

        decode_layer(layer + 1, start + half, half)

    try:
        decode_layer(0, 0, len(llr))
    except StopAtPhi:
        pass

    return result[0]


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        L = self.list_size

        paths = [{"u_hat": np.zeros(self.N, dtype=int), "pm": 0.0}]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                cur_llr = _llr_at_phi(llr_ch, path["u_hat"], phi, n)

                if self.frozen_bits[phi]:
                    path["pm"] = _pm_update(path["pm"], cur_llr, 0)
                    path["u_hat"][phi] = 0
                    expanded.append(path)
                else:
                    for u in (0, 1):
                        expanded.append({
                            "u_hat": path["u_hat"].copy(),
                            "pm": _pm_update(path["pm"], cur_llr, u),
                        })
                        expanded[-1]["u_hat"][phi] = u

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[:L]

        best = min(paths, key=lambda p: p["pm"])

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
