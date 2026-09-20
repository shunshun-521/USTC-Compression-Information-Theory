"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import (
    bit_reversal_permutation,
    map_decoded_to_u_domain,
    transform_frozen_bits,
)
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _prepare_llr,
)


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
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent_id", "branch_bit")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.parent_id = 0
        self.branch_bit = 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        info_idx = np.where(frozen_bits == 0)[0]
        self.frozen_set = set(np.where(transform_frozen_bits(frozen_bits, info_idx, N) == 1)[0])
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs_for_bit(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits_for_bit(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u 域 u_hat 与最优路径度量 pm。
        """
        llr = _prepare_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for l in self.decode_order:
            for path in paths:
                self._update_llrs_for_bit(path, l)

            new_paths = []

            if l in self.frozen_set:
                for path in paths:
                    llr_l = path.L[l, self.n]
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._path_metric_penalty(llr_l, 0)
                    self._update_bits_for_bit(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr_l = path.L[l, self.n]
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        child.pm += self._path_metric_penalty(llr_l, bit)
                        self._update_bits_for_bit(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        u_prime = best.u_hat.copy()
        u_hat = map_decoded_to_u_domain(u_prime, self.N)
        return u_hat, best.pm
