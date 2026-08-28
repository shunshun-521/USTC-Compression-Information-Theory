"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
    16: np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
    ),
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    g = CRC_POLYNOMIALS[crc_length]
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i : i + len(g)] ^= g
    return np.concatenate([info_bits, msg[len(info_bits) : len(info_bits) + crc_length]])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    g = CRC_POLYNOMIALS[crc_length]
    msg = np.asarray(bits, dtype=np.int8).copy()
    for i in range(len(bits) - crc_length):
        if msg[i] == 1:
            msg[i : i + len(g)] ^= g
    return bool(np.all(msg[-crc_length:] == 0))


class PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    path.L[j, s + 1] = btm + top if top_bit == 0 else btm - top

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = self.br[phi]
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    new_path = PathState(self.N, self.n, llr_ch)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.pm = path.pm + (0.0 if llr >= 0 else abs(llr))
                    new_path.u_hat[:] = path.u_hat
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = PathState(self.N, self.n, llr_ch)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        hard = 0 if llr >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr)
                        new_path.pm = path.pm + penalty
                        new_path.u_hat[:] = path.u_hat
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
