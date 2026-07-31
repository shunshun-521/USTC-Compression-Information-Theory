"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
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
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
    """单条 SCL 路径状态。"""

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new_path = PathState(self.N, self.n)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, bit_index):
        for stage in range(self.n - _active_llr_level(bit_index, self.n), self.n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_index, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, stage + 1] = _upper_llr(
                        path.L[j, stage], path.L[j + branch_size, stage]
                    )
                else:
                    path.L[j, stage + 1] = _lower_llr(
                        path.L[j, stage],
                        path.L[j - branch_size, stage],
                        int(path.B[j - branch_size, stage + 1]),
                    )

    def _update_bits(self, path, bit_index):
        if bit_index < self.N // 2:
            return
        for stage in range(self.n, self.n - _active_bit_level(bit_index, self.n), -1):
            block_size = 2 ** stage
            branch_size = block_size // 2
            for j in range(bit_index, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, stage - 1] = int(path.B[j, stage]) ^ int(
                        path.B[j - branch_size, stage]
                    )
                    path.B[j, stage - 1] = path.B[j, stage]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for bit_index in [_bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, bit_index)
                llr = path.L[bit_index, self.n]

                if bit_index in self.frozen:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[bit_index, self.n] = 0
                    new_path.u_hat[bit_index] = 0
                    self._update_bits(new_path, bit_index)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[bit_index, self.n] = bit
                        new_path.u_hat[bit_index] = bit
                        self._update_bits(new_path, bit_index)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
