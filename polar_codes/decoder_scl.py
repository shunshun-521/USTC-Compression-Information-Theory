"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    precompute_sc_indices,
    sc_decode,
    sc_step_bit_update,
    sc_step_llr_and_decide,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================

class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch[self.br]

        for phi in range(N):
            candidates = []
            for p_idx, path in enumerate(paths):
                bit0, llr_leaf, bit_idx = sc_step_llr_and_decide(
                    phi,
                    self.llr_layer_vec[phi],
                    path.L,
                    path.B,
                    path.u_hat,
                    self.frozen_bits,
                    N,
                    self.lambda_offset,
                )
                if self.frozen_bits[bit_idx]:
                    candidates.append((path.pm + self._pm_penalty(llr_leaf, 0), p_idx, 0, bit_idx))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr_leaf, bit), p_idx, bit, bit_idx)
                        )

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[:L]

            new_paths = []
            for new_pm, parent_idx, bit, bit_idx in survivors:
                parent = paths[parent_idx]
                child = _Path(N, n)
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.u_hat[:] = parent.u_hat
                child.pm = new_pm
                child.u_hat[bit_idx] = bit
                child.B[bit_idx, n] = bit
                sc_step_bit_update(
                    phi,
                    self.bit_layer_vec[phi],
                    child.B,
                    child.u_hat,
                    N,
                    self.lambda_offset,
                )
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_positions], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
