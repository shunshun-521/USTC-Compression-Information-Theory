"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversed, bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask_top = 1 << (crc_length - 1)
    mask_all = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & mask_top:
                reg = ((reg << 1) ^ poly) & mask_all
            else:
                reg = (reg << 1) & mask_all

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    else:
        poly = 0x8005
    reg = 0
    mask_top = 1 << (crc_length - 1)
    mask_all = (1 << crc_length) - 1
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & mask_top:
                reg = ((reg << 1) ^ poly) & mask_all
            else:
                reg = (reg << 1) & mask_all
    return reg == 0


class Path:
    """SCL 单条路径"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices) if info_indices is not None else None
        self.rev = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return abs(llr) if u_bit != hard else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev]

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        decode_order = [bit_reversed(i, self.n) for i in range(self.N)]

        for phi in decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, phi)
                llr_phi = path.L[phi, self.n]

                if phi in self.frozen_set:
                    pen = self._path_metric_penalty(llr_phi, 0)
                    path.pm += pen
                    path.B[phi, self.n] = 0
                    path.u_hat[phi] = 0
                    self._update_bits(path, phi)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = Path(self.N, self.n)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.u_hat = path.u_hat.copy()
                        p.pm = path.pm + self._path_metric_penalty(llr_phi, u_bit)
                        p.B[phi, self.n] = u_bit
                        p.u_hat[phi] = u_bit
                        self._update_bits(p, phi)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p.u_hat[self.info_indices] if self.info_indices is not None else p.u_hat
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
