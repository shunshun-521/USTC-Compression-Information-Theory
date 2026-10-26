"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _build_crc_table(poly, crc_length):
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    table = []
    for byte in range(256):
        crc = byte << (crc_length - 8) if crc_length >= 8 else byte
        crc &= mask
        for _ in range(8):
            if crc & top:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
        table.append(crc)
    return table


_CRC8_TABLE = _build_crc_table(_CRC8_POLY, 8)
_CRC16_TABLE = _build_crc_table(_CRC16_POLY, 16)


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    nbytes = (n + 7) // 8
    pad = nbytes * 8 - n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i + j])
        out.append(val)
    return bytes(out)


def _crc_value(bits, crc_length=8):
    table = _CRC8_TABLE if crc_length == 8 else _CRC16_TABLE
    mask = (1 << crc_length) - 1
    data = _bits_to_bytes(bits)
    crc = 0
    for b in data:
        if crc_length == 8:
            crc = table[crc ^ b]
        else:
            crc = ((crc << 8) ^ table[((crc >> 8) ^ b) & 0xFF]) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_value(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        calc = _crc_value(bits[:-crc_length], crc_length)
        rx = 0
        for i, b in enumerate(bits[-crc_length:]):
            rx |= int(b) << (crc_length - 1 - i)
        return calc == rx
    calc = _crc_value(bits[:-crc_length], crc_length)
    rx = 0
    for i, b in enumerate(bits[-crc_length:]):
        rx |= int(b) << (crc_length - 1 - i)
    return calc == rx


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phase_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    b = path.B[j - branch_size, s + 1]
                    if b == 0:
                        path.L[j, s + 1] = path.L[j, s] + path.L[j - branch_size, s]
                    else:
                        path.L[j, s + 1] = path.L[j, s] - path.L[j - branch_size, s]

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.phase_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr_bit, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, path.L[:, 0].copy())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr_bit, bit)
                        child.B[l, self.n] = bit
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(np.int8), best.pm
