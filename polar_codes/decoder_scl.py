"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    g_operation,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_step(reg, bit, poly, crc_length):
    """CRC LFSR 单步：移位输入，最高位为 1 时异或多项式"""
    mask = (1 << crc_length) - 1
    reg = ((reg << 1) | int(bit)) & mask
    if reg & (1 << (crc_length - 1)):
        reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: poly 0x07; CRC-16: poly 0x8005（MSB-first 比特流）
    """
    poly = _crc_poly(crc_length)
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    for _ in range(crc_length):
        reg = _crc_step(reg, 0, poly, crc_length)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC，返回 True/False"""
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    return reg == 0


class _Path:
    """SCL 单条路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for path in sorted(paths, key=lambda p: p.pm):
                info_bits = path.u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    best_crc = path
                    break

        best = best_crc if best_crc is not None else best_all
        return best.u_hat.copy(), best.pm
