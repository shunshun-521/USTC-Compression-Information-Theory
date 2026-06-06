"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        remainder = _crc8_remainder(info_bits)
        crc_bits = np.array([(remainder >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        remainder = _crc16_remainder(info_bits)
        crc_bits = np.array([(remainder >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def _crc8_remainder(info_bits):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ 0x07) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_remainder(info_bits):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ 0x8005) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "C", "pm", "u_hat", "parent", "branch_bit")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.parent = None
        self.branch_bit = 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.C[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.C[l, self.n] = bit
        path.u_hat[l] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.C[j - branch_size, s - 1] = (
                            path.C[j, s] + path.C[j - branch_size, s]
                        ) % 2
                        path.C[j, s - 1] = path.C[j, s]

    def _clone_path(self, src, branch_bit):
        dst = _Path(self.N, self.n, None)
        dst.L = src.L.copy()
        dst.C = src.C.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.parent = src
        dst.branch_bit = branch_bit
        return dst

    @staticmethod
    def _path_metric_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path.pm, llr, 0)
                    path.pm = pm
                    self._update_bits(path, l, 0)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path, bit)
                        new_path.pm = self._path_metric_update(path.pm, llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_indices] if self.info_indices is not None else p.u_hat
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
