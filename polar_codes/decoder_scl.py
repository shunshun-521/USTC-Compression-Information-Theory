"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _bit_reversed


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _pm_update(pm, llr, u):
    """路径度量：与 LLR 硬判决不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == hard else abs(llr))


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent")

    def __init__(self, N, n, llr_ch, parent=None):
        if parent is None:
            rev = bit_reversal_permutation(N)
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch[rev]
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=np.int8)
        else:
            self.L = parent.L.copy()
            self.B = parent.B.copy()
            self.pm = parent.pm
            self.u_hat = parent.u_hat.copy()
        self.parent = parent


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制 B 与 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _advance_llr(self, paths, l):
        start = self.n - _active_llr_level(l, self.n)
        for path in paths:
            for s in range(start, self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        top = path.L[j, s]
                        btm = path.L[j + branch_size, s]
                        path.L[j, s + 1] = f_operation(top, btm)
                    else:
                        btm = path.L[j, s]
                        top = path.L[j - branch_size, s]
                        top_bit = path.B[j - branch_size, s + 1]
                        path.L[j, s + 1] = g_operation(top, btm, top_bit)

    def _backprop_bits(self, path, l):
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase, l in enumerate(self.decode_order):
            self._advance_llr(paths, l)
            llr_leaf = paths[0].L[l, self.n] if len(paths) == 1 else None

            new_paths = []
            if self.frozen_bits[l]:
                for path in paths:
                    llr = path.L[l, self.n]
                    path.pm = _pm_update(path.pm, llr, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._backprop_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for u in (0, 1):
                        child = _Path(self.N, self.n, llr_ch, parent=path)
                        child.pm = _pm_update(path.pm, llr, u)
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        self._backprop_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            payload_len = len(info_positions) - self.crc_length
            valid = []
            for p in paths:
                payload = p.u_hat[info_positions[:payload_len]]
                crc_part = p.u_hat[info_positions[payload_len:]]
                if crc_check(np.concatenate([payload, crc_part]), self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat.astype(int), best.pm
