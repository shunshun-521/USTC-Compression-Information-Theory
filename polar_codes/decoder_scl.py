"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, n, N, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        new = _Path(self.L.shape[1] - 1, self.L.shape[0])
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, L = self.n, self.N, self.list_size
        paths = [_Path(n, N, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            if l in self.frozen_set:
                for path in paths:
                    self._update_llrs(path, l)
                    llr = path.L[l, n]
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
            else:
                candidates = []
                for p_idx, path in enumerate(paths):
                    self._update_llrs(path, l)
                    llr = path.L[l, n]
                    for u in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr, u), p_idx, u))

                candidates.sort(key=lambda x: x[0])
                candidates = candidates[:L]

                new_paths = []
                used_src = set()
                for new_pm, src_idx, u in candidates:
                    if src_idx not in used_src:
                        path = paths[src_idx]
                        used_src.add(src_idx)
                    else:
                        path = paths[src_idx].copy()
                    path.pm = new_pm
                    path.B[l, n] = u
                    self._update_bits(path, l)
                    new_paths.append(path)
                paths = new_paths

        crc_valid = []
        for i, path in enumerate(paths):
            u_hat = path.B[:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(i)

        if crc_valid:
            best = min(crc_valid, key=lambda i: paths[i].pm)
        else:
            best = min(range(len(paths)), key=lambda i: paths[i].pm)

        u_hat = paths[best].B[:, n].astype(int)
        return u_hat, paths[best].pm


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    """验证 L=1 的 SCL 等价于 SC"""
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
