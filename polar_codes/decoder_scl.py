"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 约束。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N, n = self.N, self.n
        frozen = self.frozen_bits

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch[self.br]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                _update_llrs(l, path.L, path.B, n, N)
                current_llr = path.L[l, n]

                if frozen[l]:
                    u = 0
                    penalty = 0.0 if current_llr >= 0 else abs(current_llr)
                    new_path = _Path(N, n)
                    new_path.pm = path.pm + penalty
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.B[l, n] = u
                    new_path.u_hat[l] = u
                    _update_bits(l, new_path.B, n, N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        llr_bit = 0 if current_llr >= 0 else 1
                        penalty = 0.0 if u == llr_bit else abs(current_llr)
                        new_path = _Path(N, n)
                        new_path.pm = path.pm + penalty
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        _update_bits(l, new_path.B, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = np.where(~frozen)[0]
            passed = [p for p in paths if crc_check(p.u_hat[info_positions], self.crc_length)]
            if passed:
                best = min(passed, key=lambda p: p.pm)

        return best.u_hat, best.pm
