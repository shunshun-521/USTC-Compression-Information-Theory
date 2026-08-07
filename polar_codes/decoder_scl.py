"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    remainder = _crc_remainder(payload, poly, crc_length)
    expected = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    actual = bits[-crc_length:]
    return np.array_equal(actual, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
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

    def _propagate_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        N, n = self.N, self.n

        class Path:
            __slots__ = ("L", "B", "pm", "u_hat")

            def __init__(self):
                self.L = np.zeros((N, n + 1), dtype=np.float64)
                self.B = np.zeros((N, n + 1), dtype=np.int32)
                self.pm = 0.0
                self.u_hat = np.zeros(N, dtype=int)

            def copy(self):
                p = Path()
                p.L = self.L.copy()
                p.B = self.B.copy()
                p.pm = self.pm
                p.u_hat = self.u_hat.copy()
                return p

        init = Path()
        init.L[:, 0] = llr_ch
        paths = [init]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(paths, l)
            new_paths = []

            for path in paths:
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._pm_penalty(llr, 0)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        child.pm += self._pm_penalty(llr, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]
            self._propagate_bits(paths, l)

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
            else:
                crc_valid.append(path)

        best = min(crc_valid if crc_valid else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
