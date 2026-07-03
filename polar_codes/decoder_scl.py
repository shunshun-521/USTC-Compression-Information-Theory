"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数。"""
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length <= 8 else 1):
            if crc_length <= 8:
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
        remainder = _crc_remainder(info_bits, poly, 8)
        crc_bits = np.array([(remainder >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= (int(bit) << 15)
            for _ in range(1):
                if reg & 0x8000:
                    reg = ((reg << 1) & 0xFFFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFFFF
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) & 0xFFFF) ^ poly
            else:
                reg = (reg << 1) & 0xFFFF
        remainder = reg
        crc_bits = np.array([(remainder >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr, bit):
    """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.pm = path.pm
        return new_path

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        B = path.B
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat（最优路径）和 pm（路径度量）。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += _pm_penalty(llr_val, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._clone_path(path)
                        p.pm += _pm_penalty(llr_val, bit)
                        p.B[l, self.n] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = p.B[:, self.n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode
    from construction import ga_construction

    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(1)
    mismatch = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatch += 1
    print(f"SCL L=1 vs SC mismatch: {mismatch}/20")
