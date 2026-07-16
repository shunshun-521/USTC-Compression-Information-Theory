"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _prepare_llr,
    f_operation,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _PathState:
    """单条 SCL 路径状态（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "parent")

    def __init__(self, N, n, parent=None):
        self.pm = 0.0
        self.parent = parent
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=int)
        else:
            self.L = parent.L
            self.B = parent.B

    def clone(self):
        child = _PathState(0, 0, parent=self)
        child.pm = self.pm
        return child

    def materialize(self, N, n):
        if self.parent is not None:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.parent = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _update_llrs(self, paths, l):
        for path in paths:
            L, B = path.L, path.B
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = g_operation(
                            L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                        )

    def _update_bits(self, path, l):
        B = path.B
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _metric_penalty(llr, u_val):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_val == hard else abs(llr)

    def decode(self, llr_ch):
        """返回 (u_hat, pm)"""
        llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n

        root = _PathState(N, n)
        root.L[:, 0] = llr
        paths = [root]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            self._update_llrs(paths, l)
            new_paths = []

            if l in self.frozen_set:
                for path in paths:
                    llr_val = path.L[l, n]
                    path.pm += self._metric_penalty(llr_val, 0)
                    path.materialize(N, n)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr_val = path.L[l, n]
                    p0 = path.clone()
                    p1 = path.clone()
                    p0.pm += self._metric_penalty(llr_val, 0)
                    p0.materialize(N, n)
                    p0.B[l, n] = 0
                    self._update_bits(p0, l)
                    new_paths.append(p0)

                    p1.pm += self._metric_penalty(llr_val, 1)
                    p1.materialize(N, n)
                    p1.B[l, n] = 1
                    self._update_bits(p1, l)
                    new_paths.append(p1)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = []
            for path in paths:
                info_bits = path.B[:, n][info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm


def verify_scl_equals_sc(N=64, frozen_bits=None, num_trials=50, seed=42):
    """验证 L=1 的 SCL 等价于 SC"""
    from decoder_sc import sc_decode
    from encoder import polar_encode

    rng = np.random.default_rng(seed)
    if frozen_bits is None:
        frozen_bits = np.zeros(N, dtype=int)
        frozen_bits[N // 2:] = 1
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        info = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        u[info] = rng.integers(0, 2, size=len(info))
        x = polar_encode(u)
        llr = (1 - 2 * x) * 100.0
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
