"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _update_bits, _update_llrs
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    top_bit = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
    """单条 SCL 路径状态。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _llr_to_bit(self, llr):
        return 0 if llr >= 0 else 1

    def _pm_penalty(self, llr, u):
        return 0.0 if u == self._llr_to_bit(llr) else abs(llr)

    def _clone_path(self, src):
        dst = PathState(self.n, self.N)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [PathState(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path.pm += self._pm_penalty(cur_llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append((new_path.pm, new_path))
                else:
                    for u_val in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._pm_penalty(cur_llr, u_val)
                        new_path.B[l, self.n] = u_val
                        new_path.u_hat[l] = u_val
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [item[1] for item in candidates[: self.list_size]]

        survivors = [(p.pm, p.u_hat.copy()) for p in paths]

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in survivors
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = min(survivors, key=lambda x: x[0])
        return best[1], best[0]
