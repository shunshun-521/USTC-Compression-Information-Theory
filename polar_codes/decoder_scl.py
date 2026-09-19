"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level
from encoder import bit_reversal_permutation


def _crc_poly_int(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    """计算 CRC 余数（MSB-first 比特流，按位寄存器）"""
    poly = _crc_poly_int(crc_length)
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        if crc & top:
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: x^8 + x^2 + x + 1 (0x07)
    CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    remainder = _crc_remainder(bits, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Vangala 置换调度 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _update_llrs(self, path, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        start_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, start_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (path.B[j, s] + path.B[j - branch_size, s]) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.pm = path.pm
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.u_hat[:] = path.u_hat
        return new_path

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            current_llr = paths[0].L[l, self.n]
            new_paths = []

            if self.frozen_bits[l]:
                for path in paths:
                    bit = 0
                    if current_llr < 0:
                        path.pm += abs(current_llr)
                    path.u_hat[l] = bit
                    path.B[l, self.n] = bit
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        llr_val = child.L[l, self.n]
                        if (bit == 0 and llr_val < 0) or (bit == 1 and llr_val >= 0):
                            child.pm += abs(llr_val)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            crc_paths = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            best = crc_paths[0] if crc_paths else paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
