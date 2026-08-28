"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_mod(bits, crc_length, poly):
    bits = np.asarray(bits, dtype=int).copy()
    n = len(bits) - crc_length
    for i in range(n):
        if bits[i] == 1:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    bits[i + j] ^= 1
    return bits


def _crc_encode_bits(info_bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    padded = np.concatenate([np.asarray(info_bits, dtype=int), np.zeros(crc_length, dtype=int)])
    remainder = _crc_mod(padded, crc_length, poly)[-crc_length:]
    val = 0
    for bit in remainder:
        val = (val << 1) | int(bit)
    return val


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    rem = _crc_mod(np.asarray(bits, dtype=int), crc_length, poly)
    return int(np.dot(rem[-crc_length:], 1 << np.arange(crc_length - 1, -1, -1)))


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_encode_bits(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC_POLYNOMIALS[crc_length]
    rem = _crc_mod(bits.copy(), crc_length, poly)
    return np.all(rem[-crc_length:] == 0)


class PathState:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, i):
        l = _bit_reversed(i, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, i, u_bit):
        l = _bit_reversed(i, self.n)
        path.B[l, self.n] = u_bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] + path.B[j - branch_size, s]
                        ) % 2
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _path_metric_penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            for path in paths:
                self._update_llrs(path, i)

            if self.frozen_bits[i]:
                for path in paths:
                    path.pm += self._path_metric_penalty(path.L[l, self.n], 0)
                    path.u_hat[i] = 0
                    self._update_bits(path, i, 0)
            else:
                new_paths = []
                for path in paths:
                    llr_leaf = path.L[l, self.n]
                    for u_bit in (0, 1):
                        child = PathState(self.N, self.n)
                        child.pm = path.pm + self._path_metric_penalty(llr_leaf, u_bit)
                        child.u_hat = path.u_hat.copy()
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat[i] = u_bit
                        self._update_bits(child, i, u_bit)
                        new_paths.append(child)
                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm


def scl_decode_channel(llr_ch, frozen_bits, list_size=4, crc_length=0):
    """信道顺序 LLR 的 SCL 译码包装。"""
    decoder = SCLDecoder(len(llr_ch), frozen_bits, list_size=list_size, crc_length=crc_length)
    return decoder.decode(llr_ch)
