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
    _reorder_channel_llrs,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    data = []
    bl = list(np.asarray(bits, dtype=int))
    while bl:
        chunk = bl[:8]
        bl = bl[8:]
        while len(chunk) < 8:
            chunk.append(0)
        byte = 0
        for b in chunk:
            byte = (byte << 1) | int(b)
        data.append(byte)
    return data


def _crc8_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    if crc_length == 8:
        crc_val = _crc8_bytes(data)
    else:
        crc_val = _crc16_bytes(data)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_bytes(data) == 0
    return _crc16_bytes(data) == 0


class Path:
    """SCL 单条路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size >> 1
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = int(path.B[j - branch_size, s + 1])
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s], path.L[j, s], top_bit
                        )

    def _update_bits(self, paths, l):
        if l < self.N / 2:
            return
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _path_metric_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        llrs = _reorder_channel_llrs(llr_ch)

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llrs

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            self._update_llrs(paths, l)
            llr = paths[0].L[l, n]

            if l in self.frozen_set:
                for path in paths:
                    llr_val = path.L[l, n]
                    path.pm += self._path_metric_penalty(llr_val, 0)
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
            else:
                candidates = []
                for pidx, path in enumerate(paths):
                    llr_val = path.L[l, n]
                    for u in (0, 1):
                        pm = path.pm + self._path_metric_penalty(llr_val, u)
                        candidates.append((pm, pidx, u))

                candidates.sort(key=lambda x: x[0])
                candidates = candidates[: self.list_size]

                new_paths = []
                for pm, pidx, u in candidates:
                    src = paths[pidx]
                    new_path = Path(N, n)
                    new_path.L = src.L.copy()
                    new_path.B = src.B.copy()
                    new_path.u_hat = src.u_hat.copy()
                    new_path.pm = pm
                    new_path.B[l, n] = u
                    new_path.u_hat[l] = u
                    new_paths.append(new_path)
                paths = new_paths

            self._update_bits(paths, l)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
