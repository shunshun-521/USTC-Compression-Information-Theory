"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    LLR_CLIP,
)

CRC8_DIVISOR = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_DIVISOR = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _gf2_remainder(msg_bits, divisor):
    """GF(2) 多项式长除法求余数。"""
    msg = list(map(int, msg_bits))
    n = len(divisor)
    for i in range(len(msg) - n + 1):
        if msg[i]:
            for j in range(n):
                msg[i + j] ^= divisor[j]
    return msg[-(n - 1) :]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    divisor = CRC8_DIVISOR if crc_length == 8 else CRC16_DIVISOR
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _gf2_remainder(extended, divisor)
    return np.concatenate([info_bits, np.array(remainder, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    divisor = CRC8_DIVISOR if crc_length == 8 else CRC16_DIVISOR
    remainder = _gf2_remainder(bits, divisor)
    return all(x == 0 for x in remainder)


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        brp = bit_reversal_permutation(self.N)
        llr_ch = np.clip(llr_ch[brp], -LLR_CLIP, LLR_CLIP)

        path = _PathState(self.N, self.n)
        path.L[:, 0] = llr_ch
        paths = [path]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path
                    new_path.pm += self._pm_penalty(llr_val, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _PathState(self.N, self.n)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.pm = path.pm + self._pm_penalty(llr_val, u_bit)
                        new_path.u_hat[:] = path.u_hat
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = paths[0]

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen.u_hat.copy(), chosen.pm
