"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """SCL 单条路径。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = Path.__new__(Path)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


def _path_update_llrs(path, phase, n, N):
    L, B = path.L, path.B
    for s in range(n - active_llr_level(phase, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(phase, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _path_update_bits(path, phase, n, N):
    B = path.B
    if phase < N // 2:
        return
    for s in range(n, n - active_bit_level(phase, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phase, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [Path(self.N, self.n, llr_ch)]

        for phase in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                _path_update_llrs(path, phase, self.n, self.N)
                llr_val = path.L[phase, self.n]

                if phase in self.frozen_set:
                    new_path = path.copy()
                    hard_bit = 0 if llr_val >= 0 else 1
                    if hard_bit != 0:
                        new_path.pm += abs(llr_val)
                    new_path.u_hat[phase] = 0
                    new_path.B[phase, self.n] = 0
                    _path_update_bits(new_path, phase, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        hard_bit = 0 if llr_val >= 0 else 1
                        if bit != hard_bit:
                            new_path.pm += abs(llr_val)
                        new_path.u_hat[phase] = bit
                        new_path.B[phase, self.n] = bit
                        _path_update_bits(new_path, phase, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm

    def _crc_pass(self, path):
        info_bits = path.u_hat[~self.frozen_bits]
        return crc_check(info_bits, self.crc_length)
