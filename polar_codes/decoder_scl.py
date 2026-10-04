"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np

from decoder_sc import _update_bits, _update_llrs
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB-first 比特流）"""
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _path_llr_at_leaf(self, path, l):
        L, B = path.L, path.B
        Lc = copy.deepcopy(L)
        Bc = copy.deepcopy(B)
        _update_llrs(Lc, Bc, l, self.n)
        return Lc[l, self.n], Lc, Bc

    def _continue_path(self, path, l, bit, Lc, Bc):
        path.L = Lc
        path.B = Bc
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n)
        path.u_hat[l] = bit

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]
        paths[0].pm = 0.0

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                llr_leaf, Lc, Bc = self._path_llr_at_leaf(path, l)
                if self.frozen_bits[l]:
                    pm = path.pm + (0.0 if llr_leaf >= 0 else abs(llr_leaf))
                    p = copy.deepcopy(path)
                    self._continue_path(p, l, 0, Lc, Bc)
                    p.pm = pm
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        pm = path.pm
                        if (bit == 0 and llr_leaf < 0) or (bit == 1 and llr_leaf >= 0):
                            pm += abs(llr_leaf)
                        p = copy.deepcopy(path)
                        self._continue_path(p, l, bit, copy.deepcopy(Lc), copy.deepcopy(Bc))
                        p.pm = pm
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_indices] if self.info_indices is not None else p.u_hat
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)
        return chosen.u_hat.copy(), chosen.pm
