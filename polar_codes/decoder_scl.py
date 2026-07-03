"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reverse,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
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
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    remainder = _crc_remainder(payload, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-crc_length + i])
    return remainder == received


class Path:
    """单条译码路径（Lazy Copy）。"""

    __slots__ = ("pm", "L_ptr", "B_ptr", "active")

    def __init__(self, pm=0.0, L_ptr=0, B_ptr=0):
        self.pm = pm
        self.L_ptr = L_ptr
        self.B_ptr = B_ptr
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = np.array(
            [_bit_reverse(i, self.n) for i in range(N)], dtype=int
        )

        self.L_pool = [np.zeros((N, self.n + 1), dtype=np.float64)]
        self.B_pool = [np.zeros((N, self.n + 1), dtype=np.int8)]

    def _alloc_arrays(self):
        self.L_pool.append(np.zeros((self.N, self.n + 1), dtype=np.float64))
        self.B_pool.append(np.zeros((self.N, self.n + 1), dtype=np.int8))
        return len(self.L_pool) - 1

    def _clone_path(self, path):
        idx = self._alloc_arrays()
        self.L_pool[idx] = self.L_pool[path.L_ptr].copy()
        self.B_pool[idx] = self.B_pool[path.B_ptr].copy()
        return Path(path.pm, idx, idx)

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        idx0 = self._alloc_arrays()
        L = self.L_pool[idx0]
        B = self.B_pool[idx0]
        L[:] = 0.0
        B[:] = 0
        L[:, 0] = llr_ch

        paths = [Path(0.0, idx0, idx0)]

        for i in range(self.N):
            l = _bit_reverse(i, self.n)
            new_paths = []
            for path in paths:
                Lp = self.L_pool[path.L_ptr]
                Bp = self.B_pool[path.B_ptr]
                _update_llrs(Lp, Bp, l, self.n, self.N)
                llr_val = Lp[l, self.n]

                if self.frozen_bits[i]:
                    child = self._clone_path(path)
                    child.pm += self._pm_penalty(llr_val, 0)
                    Lc = self.L_pool[child.L_ptr]
                    Bc = self.B_pool[child.B_ptr]
                    Bc[l, self.n] = 0
                    _update_bits(Bc, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._pm_penalty(llr_val, u_bit)
                        Lc = self.L_pool[child.L_ptr]
                        Bc = self.B_pool[child.B_ptr]
                        Bc[l, self.n] = u_bit
                        _update_bits(Bc, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = []
            for p in paths:
                u_cand = self.B_pool[p.B_ptr][self.br, self.n].astype(int)
                if crc_check(u_cand[info_mask], self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = self.B_pool[best.B_ptr][self.br, self.n].astype(int)
        return u_hat, best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)

    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + np.random.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC mismatches: {mismatches}/20")
