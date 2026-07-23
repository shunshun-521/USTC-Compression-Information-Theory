"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _f_boxplus,
    _g_boxplus,
    _reorder_channel_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ('pm', 'L', 'B', 'active')

    def __init__(self, N, n, llr_natural):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_natural
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _f_boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _g_boxplus(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, src):
        dst = object.__new__(_Path)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.active = True
        return dst

    def decode(self, llr_ch):
        llr_nat = _reorder_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr_nat)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    candidates.append((path.pm, u_hat))
            else:
                candidates.append((path.pm, u_hat))

        if not candidates:
            best = min(paths, key=lambda p: p.pm)
            return best.B[:, self.n].astype(int), best.pm

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm


if __name__ == '__main__':
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)

    for L in [1, 4]:
        mism = errors = 0
        for _ in range(50):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            sigma = eb_n0_to_sigma(8.0, K / N)
            llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
            uh_sc = sc_decode(llr, frozen_bits)
            uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=L).decode(llr)
            if L == 1 and not np.array_equal(uh_sc, uh_scl):
                mism += 1
            if not np.array_equal(uh_scl[info_idx], u[info_idx]):
                errors += 1
        print(f'L={L} SC-SCL mismatch: {mism}/50, frame errors: {errors}/50')
