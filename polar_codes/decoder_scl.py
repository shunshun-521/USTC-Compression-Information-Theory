"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _prepare_channel_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_exact,
    _lower_llr_exact,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
_CRC16_POLY = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=int)
    for bit in info_bits:
        feedback = reg[-1] ^ bit
        reg[1:] = reg[:-1]
        reg[0] = 0
        if feedback:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _pm_penalty(llr, u):
    """路径度量惩罚：与 LLR 不一致时加 |LLR|"""
    hard = _llr_to_bit(llr)
    return 0.0 if u == hard else abs(llr)


class _Path:
    """单条 SCL 路径（Lazy Copy：共享 LLR/比特数组引用）"""
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = np.clip(llr_ch, -1e20, 1e20)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SC + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(path.B[j - branch_size, s + 1]) else int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = _lower_llr_exact(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bi = int(path.B[j, s]) if not np.isnan(path.B[j, s]) else 0
                    bj = int(path.B[j - branch_size, s]) if not np.isnan(path.B[j - branch_size, s]) else 0
                    path.B[j - branch_size, s - 1] = bi ^ bj
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += _pm_penalty(cur_llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += _pm_penalty(cur_llr, u)
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
