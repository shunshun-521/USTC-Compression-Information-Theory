"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
    """单条 SCL 路径状态（Lazy Copy）。"""

    __slots__ = ("pm", "u_hat", "L", "B", "shared")

    def __init__(self, N, n, source=None):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        if source is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=int)
            self.shared = False
        else:
            self.L = source.L
            self.B = source.B
            self.shared = True

    def copy_on_write(self):
        if self.shared:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.shared = False


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u_val):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_val == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        init = PathState(N, n)
        init.L[:, 0] = llr_ch
        paths = [init]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                path.copy_on_write()
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if self.frozen_bits[phi]:
                    new_path = PathState(N, n, source=path)
                    new_path.copy_on_write()
                    new_path.pm = path.pm + self._pm_penalty(llr, 0)
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[phi] = 0
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = PathState(N, n, source=path)
                        new_path.copy_on_write()
                        new_path.pm = path.pm + self._pm_penalty(llr, u_val)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[phi] = u_val
                        new_path.B[l, n] = u_val
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
