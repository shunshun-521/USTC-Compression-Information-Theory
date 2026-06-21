"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB 先行）。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    penalty = 0.0 if u == hard else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, pm, L, B, u_hat):
        self.pm = pm
        self.L = L
        self.B = B
        self.u_hat = u_hat

    def copy(self):
        return _Path(self.pm, self.L.copy(), self.B.copy(), self.u_hat.copy())


class SCLDecoder:
    """SCL 译码器（mcba1n 风格相位顺序）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phase_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _new_path(self, llr):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = llr
        return _Path(0.0, L, B, np.zeros(self.N, dtype=int))

    def _compute_llr(self, path, phi):
        l = self.phase_order[phi]
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch, s])
                else:
                    top_bit = path.B[j - branch, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch, s], path.L[j, s], top_bit
                    )

    def _propagate_bits(self, path, phi):
        l = self.phase_order[phi]
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = path.B[j, s] ^ path.B[j - branch, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._compute_llr(path, phi)
                l = self.phase_order[phi]
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    new_path.pm = _path_metric_update(path.pm, llr_val, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._propagate_bits(new_path, phi)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = _path_metric_update(path.pm, llr_val, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._propagate_bits(child, phi)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
