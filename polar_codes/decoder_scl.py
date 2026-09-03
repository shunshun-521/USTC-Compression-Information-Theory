"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
    upper_llr_exact,
    lower_llr_exact,
)
from encoder import bit_reversal_permutation


# CRC-8 polynomial 0x07 (x^8 + x^2 + x + 1)
CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int)
    nbytes = (len(bits) + 7) // 8
    data = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b:
            data[i // 8] |= 1 << (7 - i % 8)
    return data


def _crc8_run(data_bytes):
    crc = 0
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_run(data_bytes):
    crc = 0
    for byte in data_bytes:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    if crc_length == 8:
        crc_val = _crc8_run(data)
        crc_bits = np.array([(crc_val >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        crc_val = _crc16_run(data)
        crc_bits = np.array([(crc_val >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_run(data) == 0
    return _crc16_run(data) == 0


class Path:
    """单条 SCL 译码路径（Lazy Copy）。"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _init_paths(self, llr_perm):
        path = Path(self.N, self.n)
        path.L[:, 0] = llr_perm
        return [path]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr_exact(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _copy_path(self, path):
        new_path = Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        br = bit_reversal_permutation(self.N)
        ibr = np.argsort(br)
        llr_perm = llr_ch[ibr]

        paths = self._init_paths(llr_perm)

        for phi_nat in range(self.N):
            l = bit_reversed(phi_nat, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    if llr_val < 0:
                        path.pm += abs(llr_val)
                    path.B[l, self.n] = bit
                    path.u_hat[l] = bit
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.B[l, self.n] = bit
                        p.u_hat[l] = bit
                        if bit == 0 and llr_val < 0:
                            p.pm += abs(llr_val)
                        elif bit == 1 and llr_val >= 0:
                            p.pm += abs(llr_val)
                        self._update_bits(p, l)
                        new_paths.append(p)

            if len(new_paths) > self.list_size:
                new_paths.sort(key=lambda p: p.pm)
                new_paths = new_paths[:self.list_size]

            paths = new_paths

        # 选择最优路径
        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        if crc_pass:
            best = min(crc_pass, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
