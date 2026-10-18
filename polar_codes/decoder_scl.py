"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from channel import reorder_llr_for_decoder
from decoder_sc import _SCD, _bit_reversed, _active_llr_level, _active_bit_level, _upper_llr, _lower_llr

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for b in bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg
    if crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for b in bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    return _crc_remainder(bits, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（基于 PSCD 的多路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = path.L[j, s]
                    btm_llr = path.L[j + branch_size, s]
                    path.L[j, s + 1] = _upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = path.L[j, s]
                    top_llr = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (path.B[j, s] ^ path.B[j - branch_size, s]) % 2
                    path.B[j - branch_size, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, np.isin(np.arange(self.N), list(self.frozen)))
            return u_hat, 0.0

        llr_ch = reorder_llr_for_decoder(np.asarray(llr_ch, dtype=np.float64), self.N)
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, self.n] = llr_ch

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pidx, 0))
                else:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pidx, 0))
                    candidates.append((path.pm + self._pm_penalty(llr, 1), pidx, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, pidx, u_bit in candidates:
                parent = paths[pidx]
                child = _PathState(self.N, self.n)
                child.pm = new_pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = 0 if l in self.frozen else u_bit
                child.u_hat[l] = child.B[l, self.n]
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        best = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
