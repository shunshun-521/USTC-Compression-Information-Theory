"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed, _active_llr_level, _active_bit_level,
    _boxplus, _lower_llr, _frozen_set_from_mask,
)


_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC_POLY[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        fb = (reg >> (crc_length - 1)) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC_POLY[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = (reg >> (crc_length - 1)) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg == 0


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_br):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_br.copy()
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = _frozen_set_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices) if info_indices is not None else None
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    btm_llr = path.L[j, s]
                    top_llr = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _crc_pass(self, u_hat):
        if self.crc_length <= 0 or self.info_indices is None:
            return True
        bits = u_hat[self.info_indices]
        return crc_check(bits, self.crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_br)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen:
                    new_path = _Path(self.N, self.n, path.L[:, 0])
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.u_hat[:] = path.u_hat
                    new_path.pm = path.pm + self._pm_penalty(cur_llr, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = _Path(self.N, self.n, path.L[:, 0])
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.u_hat[:] = path.u_hat
                        new_path.pm = path.pm + self._pm_penalty(cur_llr, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best = paths[0]
        for p in paths:
            if self._crc_pass(p.u_hat):
                if best_crc is None or p.pm < best_crc.pm:
                    best_crc = p
        chosen = best_crc if best_crc is not None else best
        return chosen.u_hat.copy(), chosen.pm
