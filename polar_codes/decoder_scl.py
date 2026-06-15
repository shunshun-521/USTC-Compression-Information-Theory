"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _bit_reversed, f_operation, g_operation
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg == 0


class Path:
    """单条译码路径。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, phi):
        L = path.L
        B = path.B
        n = self.n
        N = self.N
        l = _bit_reversed(phi, n)
        for stage in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (stage + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = g_operation(
                        L[j - branch_size, stage],
                        L[j, stage],
                        B[j - branch_size, stage + 1],
                    )

    def _update_bits(self, path, phi, bit):
        B = path.B
        n = self.n
        N = self.N
        l = _bit_reversed(phi, n)
        B[l, n] = bit
        if l < N / 2:
            return
        for stage in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << stage
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, stage - 1] = B[j, stage] ^ B[j - branch_size, stage]
                    B[j, stage - 1] = B[j, stage]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _fork(self, path):
        new_path = Path(self.N, self.n)
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.pm = path.pm
        new_path.u_hat[:] = path.u_hat
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, phi)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._fork(path)
                    new_path.pm += self._pm_penalty(llr_bit, 0)
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, phi, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._fork(path)
                        new_path.pm += self._pm_penalty(llr_bit, bit)
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, phi, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = paths[0]
        for path in paths:
            if path.pm < best_all.pm:
                best_all = path
            if self.crc_length > 0:
                info_bits = path.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length) and (
                    best_crc is None or path.pm < best_crc.pm
                ):
                    best_crc = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen.u_hat.copy(), chosen.pm
