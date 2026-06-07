"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    f_operation,
    g_operation,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_update(reg, bit, crc_length, poly):
    reg ^= int(bit) << (crc_length - 1)
    mask = (1 << crc_length) - 1
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "parent", "_owned")

    def __init__(self, pm=0.0, L=None, B=None, parent=None):
        self.pm = pm
        self.L = L
        self.B = B
        self.parent = parent
        self._owned = L is not None

    def ensure_owned(self):
        if not self._owned:
            self.L = self.parent.L.copy()
            self.B = self.parent.B.copy()
            self._owned = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy，与 SC 相同的比特倒序因子图）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _llr_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, path, l):
        path.ensure_owned()
        L, B = path.L, path.B
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        path.ensure_owned()
        B = path.B
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _apply_bit(self, path, l, u_bit):
        path.ensure_owned()
        path.B[l, self.n] = u_bit
        self._update_bits(path, l)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths = [_Path(pm=0.0, L=L0, B=B0)]

        for i in range(N):
            l = _bit_reversed_scalar(i, n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = _Path(pm=path.pm + self._llr_penalty(llr, 0), parent=path)
                    self._apply_bit(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = _Path(
                            pm=path.pm + self._llr_penalty(llr, u), parent=path
                        )
                        self._apply_bit(new_path, l, u)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        valid_paths = paths
        if self.crc_length > 0:
            crc_ok = []
            for path in paths:
                u_hat = path.B[:, n].astype(int)
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(path)
            if crc_ok:
                valid_paths = crc_ok

        best = min(valid_paths, key=lambda p: p.pm)
        u_hat = best.B[:, n].astype(int)
        for l in range(N):
            if np.isnan(best.B[l, n]):
                u_hat[l] = 0
        return u_hat, best.pm
