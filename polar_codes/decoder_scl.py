"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _align_channel_llr,
    partial_summation,
    sc_decode_recursive,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def _crc_step(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    reg &= mask
    if reg & msb:
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（系统型）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_step(reg, b, poly, crc_length)
    for _ in range(crc_length):
        reg = _crc_step(reg, 0, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent_L", "parent_B")

    def __init__(self, m, N):
        self.L = [np.zeros(N, dtype=np.float64) for _ in range(m + 1)]
        self.B = [np.zeros(N, dtype=np.int8) for _ in range(m + 1)]
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.parent_L = None
        self.parent_B = None


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new_p = _Path(self.m, self.N)
        new_p.L = path.L
        new_p.B = path.B
        new_p.pm = path.pm
        new_p.u_hat = path.u_hat.copy()
        new_p.parent_L = path
        new_p.parent_B = path
        return new_p

    def _ensure_own_arrays(self, path):
        if path.parent_L is not None:
            path.L = [arr.copy() for arr in path.L]
            path.B = [arr.copy() for arr in path.B]
            path.parent_L = None
            path.parent_B = None

    def _update_llr(self, path, phi):
        self._ensure_own_arrays(path)
        L = path.L
        for layer in range(self.m - 1, -1, -1):
            step = 1 << layer
            if (phi >> layer) & 1 == 0:
                idx = (phi // step) * step
                j = phi % step
                L[layer][idx + j] = f_operation(
                    L[layer + 1][idx + j], L[layer + 1][idx + j + step]
                )
            else:
                block = (phi // (2 * step)) * 2 * step
                j = phi % step
                c_left = partial_summation(path.u_hat[block : block + step])
                L[layer][block + j] = g_operation(
                    L[layer + 1][block + j],
                    L[layer + 1][block + j + step],
                    c_left[j],
                )

    def _propagate_bits(self, path, phi, u_bit):
        self._ensure_own_arrays(path)
        B = path.B
        B[0][0] = u_bit
        for layer in range(self.m):
            if (phi >> layer) & 1:
                step = 1 << layer
                base = phi - (phi % (2 * step))
                off = phi % step
                B[layer + 1][base + off] = B[layer][base]
                B[layer + 1][base + off + step] = B[layer][base] ^ u_bit

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat.astype(int), 0.0

        llr_ch = _align_channel_llr(llr_ch)
        paths = []
        p0 = _Path(self.m, self.N)
        p0.L[self.m][:] = llr_ch
        paths.append(p0)

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llr(path, phi)
                llr0 = path.L[0][phi]

                if self.frozen_bits[phi]:
                    u = 0
                    new_p = self._copy_path(path)
                    new_p.pm += self._pm_penalty(llr0, u)
                    new_p.u_hat[phi] = u
                    self._propagate_bits(new_p, phi, u)
                    candidates.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = self._copy_path(path)
                        new_p.pm += self._pm_penalty(llr0, u)
                        new_p.u_hat[phi] = u
                        self._propagate_bits(new_p, phi, u)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
