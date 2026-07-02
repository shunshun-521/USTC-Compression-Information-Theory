"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _path_metric_penalty(llr, u):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "parent", "lazy_L", "lazy_B")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.parent = None
        self.lazy_L = None
        self.lazy_B = None

    def copy_from(self, other):
        """Lazy copy：仅记录父路径引用。"""
        self.parent = other
        self.lazy_L = None
        self.lazy_B = None
        self.pm = other.pm

    def resolve_L(self):
        if self.lazy_L is not None:
            return self.lazy_L
        if self.parent is not None:
            return self.parent.L
        return self.L

    def resolve_B(self):
        if self.lazy_B is not None:
            return self.lazy_B
        if self.parent is not None:
            return self.parent.B
        return self.B

    def materialize(self):
        if self.parent is not None:
            self.L = self.parent.L.copy()
            self.B = self.parent.B.copy()
            self.parent = None
            self.lazy_L = None
            self.lazy_B = None


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        L = path.resolve_L()
        B = path.resolve_B()
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        B = path.resolve_B()
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        root = _Path(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths = [root]

        for l in self.decode_order:
            for p in paths:
                p.materialize()
                self._update_llrs(p, l)

            new_paths = []
            for p in paths:
                p.materialize()
                llr = p.L[l, self.n]
                if self.frozen_bits[l]:
                    pen = _path_metric_penalty(llr, 0)
                    p.pm += pen
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        child = _Path(self.N, self.n)
                        child.copy_from(p)
                        child.materialize()
                        child.pm = p.pm + _path_metric_penalty(llr, u)
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for p in paths:
            p.materialize()
            u_hat = p.B[:, self.n].astype(int)
            pm = p.pm
            if self.crc_length > 0:
                info_mask = ~self.frozen_bits
                info_bits = u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[1]:
                        best_crc = (u_hat, pm)
            if best_pm is None or pm < best_pm[1]:
                best_pm = (u_hat, pm)

        if best_crc is not None:
            return best_crc[0], best_crc[1]
        return best_pm[0], best_pm[1]
