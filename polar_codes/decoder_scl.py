"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Vangala 2014 置换 SC 结构
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
    _update_llrs,
    _update_bits,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [
            _bit_reversed_index(i, self.n) for i in range(N)
        ]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        u_from_llr = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch[br]

        for phase, l in enumerate(self.decode_order):
            candidates = []
            for p_idx, path in enumerate(paths):
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_val = path.L[l, self.n]
                if l in self.frozen_set:
                    candidates.append(
                        (path.pm + self._pm_penalty(llr_val, 0), p_idx, 0)
                    )
                else:
                    for u_bit in (0, 1):
                        candidates.append(
                            (
                                path.pm + self._pm_penalty(llr_val, u_bit),
                                p_idx,
                                u_bit,
                            )
                        )

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, src_idx, u_bit in selected:
                src = paths[src_idx]
                new_path = _Path(self.N, self.n)
                new_path.L = src.L.copy()
                new_path.B = src.B.copy()
                new_path.u_hat = src.u_hat.copy()
                new_path.pm = pm
                new_path.B[l, self.n] = u_bit
                new_path.u_hat[l] = u_bit
                _update_bits(new_path.B, l, self.n, self.N)
                new_paths.append(new_path)
            paths = new_paths

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
