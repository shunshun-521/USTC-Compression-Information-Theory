"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import _bit_reversed, _update_bits, _update_llrs


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, poly, crc_length):
    """MSB-first CRC remainder."""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for bit in np.asarray(bits, dtype=int):
        reg <<= 1
        if bit:
            reg |= 1
        if reg & top:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: poly 0x07 (x^8 + x^2 + x + 1), MSB-first
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        return pm + abs(llr)
    return pm


class Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        N = self.N
        n = self.n

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    child = Path(N, n)
                    child.pm = _pm_update(path.pm, llr, 0)
                    child.u_hat = path.u_hat.copy()
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    _update_bits(child.B, l, n, N)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = Path(N, n)
                        child.pm = _pm_update(path.pm, llr, u)
                        child.u_hat = path.u_hat.copy()
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.B[l, n] = u
                        child.u_hat[l] = u
                        _update_bits(child.B, l, n, N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        pool = crc_pass if crc_pass else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat, best.pm
