"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


def _crc_generator(crc_length):
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 0, 1, 1, 1]
    if crc_length == 16:
        return [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(bits, crc_length):
    gen = _crc_generator(crc_length)
    msg = [int(b) for b in bits]
    n = len(gen)
    for i in range(len(msg) - n + 1):
        if msg[i] == 1:
            for j in range(n):
                msg[i + j] ^= gen[j]
    return msg[-(n - 1):]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]),
        crc_length,
    )
    return np.concatenate([info_bits, np.array(rem, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    rem = _crc_remainder(bits, crc_length)
    return all(r == 0 for r in rem)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(self.frozen_bits == 1)[0])
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)

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
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.brp].copy()

        init = _Path(self.N, self.n)
        init.L[:, 0] = llr
        paths = [init]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_idx:
                    new_path = _Path(self.N, self.n)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = path.pm
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    if cur_llr < 0:
                        new_path.pm += abs(cur_llr)
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        hard = 0 if cur_llr >= 0 else 1
                        if bit != hard:
                            new_path.pm += abs(cur_llr)
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            crc_pass = [p for p in paths if self._crc_pass(p)]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        return best.u_hat, best.pm

    def _crc_pass(self, path):
        info_bits = path.u_hat[self.info_idx]
        return crc_check(info_bits, self.crc_length)
