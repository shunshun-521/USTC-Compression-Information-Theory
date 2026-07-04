"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bitwise(bits, poly, crc_length):
    """按位输入、每比特一次移位的 CRC"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def _solve_crc_bits(info_bits, poly, crc_length):
    """求解 CRC 校验位，使 CRC(info||crc)==0"""
    base = list(info_bits) + [0] * crc_length
    syndrome = _crc_bitwise(base, poly, crc_length)
    if syndrome == 0:
        return [0] * crc_length

    def to_bits(value):
        return np.array(
            [(value >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
            dtype=np.uint8,
        )

    columns = []
    for i in range(crc_length):
        test = list(info_bits) + [0] * crc_length
        test[len(info_bits) + i] = 1
        effect = _crc_bitwise(test, poly, crc_length) ^ syndrome
        columns.append(to_bits(effect))

    mat = np.stack(columns, axis=1)  # crc_length x crc_length
    rhs = to_bits(syndrome)

    for col in range(crc_length):
        pivot = None
        for row in range(col, crc_length):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != col:
            mat[[col, pivot]] = mat[[pivot, col]]
            rhs[[col, pivot]] = rhs[[pivot, col]]
        for row in range(crc_length):
            if row != col and mat[row, col]:
                mat[row] ^= mat[col]
                rhs[row] ^= rhs[col]

    solution = [0] * crc_length
    for i in range(crc_length):
        if mat[i, i]:
            solution[i] = int(rhs[i])
    return solution


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = np.array(_solve_crc_bits(info_bits, poly, crc_length), dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_bitwise(bits, poly, crc_length) == 0


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan)
    self.L[:, 0] = llr_ch
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
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
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = _bit_reversed_index(phase, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr_leaf, 0)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr_ch)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm
                        new_path.u_hat = path.u_hat.copy()

                        penalty = self._path_metric_penalty(llr_leaf, bit)
                        new_path.pm += penalty
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.astype(int), chosen.pm
