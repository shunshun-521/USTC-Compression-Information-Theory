"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _build_tree,
    _llr_at_phase,
    _load_channel_llrs,
    clone_tree,
)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("root", "u_hat", "pm")

    def __init__(self, root, u_hat, pm=0.0):
        self.root = root
        self.u_hat = u_hat
        self.pm = pm

    def clone(self):
        return _Path(clone_tree(self.root), self.u_hat.copy(), self.pm)


class SCLDecoder:
    """SCL 译码器（树结构 + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self._template = _build_tree(N, frozen_bits)

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        root = clone_tree(self._template)
        _load_channel_llrs(root, llr_ch)
        paths = [_Path(root, np.zeros(self.N, dtype=int), 0.0)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr = _llr_at_phase(path.root, phi, path.u_hat)

                if self.frozen_bits[phi]:
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = path.clone()
                        p.pm += self._pm_penalty(llr, bit)
                        p.u_hat[phi] = bit
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
