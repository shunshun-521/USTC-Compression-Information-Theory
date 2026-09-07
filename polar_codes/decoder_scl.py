"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_bits(message_bits):
    crc = 0
    for bit in message_bits:
        msb = (crc >> 7) & 1
        crc = (crc << 1) & 0xFF
        if int(bit) ^ msb:
            crc ^= CRC8_POLY
    return crc


def _crc16_bits(message_bits):
    crc = 0
    for bit in message_bits:
        msb = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if int(bit) ^ msb:
            crc ^= CRC16_POLY
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        remainder = _crc8_bits(info_bits)
        fmt = "08b"
    else:
        remainder = _crc16_bits(info_bits)
        fmt = "016b"
    crc_bits = np.array([int(x) for x in format(remainder, fmt)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        return _crc8_bits(bits) == 0
    return _crc16_bits(bits) == 0


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1)
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = _upper_llr(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        path.L[j, s + 1] = _lower_llr(
                            path.L[j, s],
                            path.L[j - branch_size, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _propagate_bits(self, path, l):
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

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(paths, l)
            new_paths = []

            for path in paths:
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    p = path.copy()
                    p.pm += self._pm_penalty(llr, 0)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._propagate_bits(p, l)
                    new_paths.append(p)
                else:
                    for u_cand in (0, 1):
                        p = path.copy()
                        p.pm += self._pm_penalty(llr, u_cand)
                        p.u_hat[l] = u_cand
                        p.B[l, self.n] = u_cand
                        self._propagate_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.u_hat, best.pm
