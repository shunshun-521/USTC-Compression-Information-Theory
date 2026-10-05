"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, r):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (r - 1))
        for _ in range(8):
            if reg & (1 << (r - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << r) - 1)
            else:
                reg = (reg << 1) & ((1 << r) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("pm", "u_hat", "active")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（L=1 时退化为 SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N) for _ in range(self.L)]
        paths[0].active = True

        for phi in range(self.N):
            candidates = []
            for p in paths:
                if not p.active:
                    continue
                llr_phi = self._bit_llr(llr_ch, p.u_hat, phi)
                if self.frozen_bits[phi]:
                    pen = abs(llr_phi) if llr_phi < 0 else 0.0
                    new = _Path(self.N)
                    new.u_hat = p.u_hat.copy()
                    new.u_hat[phi] = 0
                    new.pm = p.pm + pen
                    candidates.append(new)
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (bit == 0 and llr_phi >= 0) or (bit == 1 and llr_phi < 0) else abs(llr_phi)
                        new = _Path(self.N)
                        new.u_hat = p.u_hat.copy()
                        new.u_hat[phi] = bit
                        new.pm = p.pm + pen
                        candidates.append(new)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.L]
            while len(paths) < self.L:
                paths.append(_Path(self.N))

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x.pm)
        return best.u_hat, best.pm

    def _bit_llr(self, llr_ch, u_partial, phi):
        """用逐层 f/g 计算当前比特 LLR（简化实现）。"""
        llr = llr_ch.copy()
        n = int(np.log2(self.N))

        def walk(node_llr, depth, offset):
            if depth == n:
                return node_llr[0]
            half = len(node_llr) // 2
            left = f_operation(node_llr[:half], node_llr[half:])
            if phi < offset + half:
                return walk(left, depth + 1, offset)
            u_left = u_partial[offset:offset + half]
            right = g_operation(node_llr[:half], node_llr[half:], u_left)
            return walk(right, depth + 1, offset + half)

        return walk(llr, 0, 0)
