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
    f_operation_exact,
    g_operation_exact,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_step(reg, bit, crc_length, poly):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, crc_length, poly)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, crc_length, poly)
    return reg == 0


def _pm_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent", "branch_bit")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.parent = None
        self.branch_bit = None


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制数组引用，裁剪时再深拷贝）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation_exact(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for step, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm = _pm_update(path.pm, llr, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = self._clone_path(path)
                        child.pm = _pm_update(child.pm, llr, u)
                        child.B[l, self.n] = u
                        child.u_hat[l] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                crc_ok.append(crc_check(info_bits, self.crc_length))
            else:
                crc_ok.append(True)

        if any(crc_ok):
            candidates = [p for p, ok in zip(paths, crc_ok) if ok]
        else:
            candidates = paths

        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
