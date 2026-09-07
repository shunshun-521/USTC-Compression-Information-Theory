"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    sc_decode_core,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ('pm', 'L', 'B', 'u_hat', 'active')

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr.copy()
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = None

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新"""
        hard = 0 if llr >= 0 else 1
        if hard == bit:
            return pm
        return pm + abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(top, btm, top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, src):
        dst = _Path(self.N, self.n, src.L[:, 0])
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_core(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue

                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    new_pm = self._path_metric_update(path.pm, llr, bit)
                    path.pm = new_pm
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm = self._path_metric_update(path.pm, llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            if candidates:
                candidates.sort(key=lambda p: p.pm)
                frozen_paths = [p for p in paths if p.active and l in self.frozen_set]
                merged = frozen_paths + candidates
                merged.sort(key=lambda p: p.pm)
                paths = merged[: self.list_size]
                for p in paths:
                    p.active = True

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for path in paths:
                payload = path.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    return path.u_hat.astype(np.int8), path.pm

        best = paths[0]
        return best.u_hat.astype(np.int8), best.pm
