"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 / CRC-16，返回信息比特 + CRC。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for byte in info_bits:
        reg ^= int(byte) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def sc_llr_at_phase(llr_ch, u_hat, phi, N):
    """计算 SC 在第 phi 个比特处的根节点 LLR。"""
    llr = llr_ch.astype(np.float64).copy()

    def walk(offset, length, bit_pos):
        if length == 1:
            return llr[offset]
        half = length // 2
        for i in range(half):
            llr[offset + i] = f_operation(llr[offset + i], llr[offset + half + i])
        if phi < bit_pos + half:
            return walk(offset, half, bit_pos)
        for i in range(half):
            llr[offset + i] = g_operation(
                llr[offset + i], llr[offset + half + i], u_hat[bit_pos + i]
            )
        return walk(offset + half, half, bit_pos + half)

    return walk(0, N, 0)


def _llr_penalty(llr, bit):
    ok = (bit == 0 and llr >= 0) or (bit == 1 and llr < 0)
    return 0.0 if ok else abs(llr)


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径仅复制 u_hat / PM，LLR 共享）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = sc_llr_at_phase(llr_ch, path.u_hat, phi, self.N)
                if self.frozen_bits[phi]:
                    bit = 0
                    path.pm += _llr_penalty(llr_phi, bit)
                    path.u_hat[phi] = bit
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N)
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + _llr_penalty(llr_phi, bit)
                        child.u_hat[phi] = bit
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        return self._select_best(paths)

    def _select_best(self, paths):
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passed = []
            for p in paths:
                payload = p.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    passed.append(p)
            if passed:
                best = min(passed, key=lambda p: p.pm)
                return best.u_hat, best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat, best.pm
