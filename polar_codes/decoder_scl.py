"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        child = _Path(self.L.shape[0], self.L.shape[1] - 1)
        child.pm = self.pm
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B/u_hat）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, path, l):
        N = self.N
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )

    def _update_bits(self, path, l):
        N = self.N
        n = self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [_Path(N, n, llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = float(path.L[l, n])

                if l in self.frozen_set:
                    path.pm = self._pm_update(path.pm, llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = self._pm_update(path.pm, llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
