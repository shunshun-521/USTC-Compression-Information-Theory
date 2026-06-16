"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _crc_polynomial(crc_length)
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_init.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            None if info_indices is None else np.asarray(info_indices, dtype=int)
        )
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(br)

    def _branch_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_init = llr_ch[self.inv_br]

        paths = [_Path(N, n, llr_init)]

        for step, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr_val = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = _Path(N, n, llr_init)
                    new_path.pm = path.pm + self._branch_penalty(llr_val, 0)
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[l] = 0
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(N, n, llr_init)
                        new_path.pm = path.pm + self._branch_penalty(llr_val, u_bit)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = u_bit
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.B[l, n] = u_bit
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            check_idx = (
                self.info_indices
                if self.info_indices is not None
                else np.where(~self.frozen_bits)[0]
            )
            valid = [
                p for p in paths if crc_check(p.u_hat[check_idx], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
