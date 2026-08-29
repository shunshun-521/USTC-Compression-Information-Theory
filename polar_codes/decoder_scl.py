"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    upper_llr, lower_llr, bit_reversed_index,
    active_llr_level, active_bit_level, sc_decode,
)
from channel import prepare_channel_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float32)
        self.B = np.zeros((N, n + 1), dtype=np.float32)


def _update_llrs_path(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n):
    N = B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s] or 0) ^ int(B[j - branch_size, s] or 0)
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n, L_size = self.N, self.n, self.list_size
        llr_init = prepare_channel_llr(llr_ch, N)

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_init

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []

            for pidx, path in enumerate(paths):
                _update_llrs_path(path.L, path.B, l, n)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    path.pm += penalty
                    path.B[l, n] = 0
                    _update_bits_path(path.B, l, n)
                    candidates.append((path.pm, pidx, None))
                else:
                    for bit in (0, 1):
                        pm_new = path.pm
                        hard = 0 if llr_val >= 0 else 1
                        if bit != hard:
                            pm_new += abs(llr_val)
                        candidates.append((pm_new, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            pool = [Path(N, n) for _ in range(L_size)]

            for rank, (pm_new, pidx, bit) in enumerate(candidates[:L_size]):
                if bit is None:
                    paths[pidx].pm = pm_new
                    dst = paths[pidx]
                else:
                    dst = pool[rank]
                    dst.pm = pm_new
                    dst.L[:] = paths[pidx].L
                    dst.B[:] = paths[pidx].B
                    dst.B[l, n] = bit
                    _update_bits_path(dst.B, l, n)
                new_paths.append(dst)

            while len(new_paths) < L_size:
                new_paths.append(pool[len(new_paths)])

            paths = new_paths[:L_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.B[:, n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.B[:, n].astype(int), path.pm

        return paths[0].B[:, n].astype(int), paths[0].pm
