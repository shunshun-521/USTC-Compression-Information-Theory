"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("pm", "u_hat", "active")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr(self, llr_ch, u_prefix, phi):
        """计算路径在相位 phi 的根 LLR（递归）"""

        def rec(L, stage, pos, target):
            if stage == 0:
                return float(L[0]) if pos == target else None
            half = 1 << (stage - 1)
            if target < pos + half:
                return rec(
                    f_operation(L[:half], L[half : 2 * half]), stage - 1, pos, target
                )
            up = u_prefix[pos : pos + half]
            return rec(
                g_operation(L[:half], L[half : 2 * half], up),
                stage - 1,
                pos + half,
                target,
            )

        return rec(llr_ch, self.n, 0, phi)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                L = self._path_llr(llr_ch, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    pen = abs(L) if L < 0 else 0.0
                    path.pm += pen
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N)
                        p.u_hat = path.u_hat.copy()
                        p.pm = path.pm
                        p.u_hat[phi] = bit
                        if (bit == 0 and L < 0) or (bit == 1 and L >= 0):
                            p.pm += abs(L)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat, best.pm
