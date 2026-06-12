"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


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
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, L, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        init_L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        init_B = np.zeros((self.N, self.n + 1), dtype=np.int32)
        init_L[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": init_L, "B": init_B, "u_hat": np.zeros(self.N, dtype=int)})

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    candidates.append((pm, p_idx, 0))
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, u)
                        candidates.append((pm, p_idx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, p_idx, u in candidates:
                src = paths[p_idx]
                dst = {
                    "pm": pm,
                    "L": src["L"].copy(),
                    "B": src["B"].copy(),
                    "u_hat": src["u_hat"].copy(),
                }
                dst["u_hat"][l] = u
                dst["B"][l, self.n] = u
                self._update_bits(dst["L"], dst["B"], l)
                new_paths.append(dst)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_scl, u_sc), "SCL L=1 != SC"


if __name__ == "__main__":
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    enc = crc_encode(bits[:4], 8)
    assert crc_check(enc, 8)
    verify_scl_equals_sc()
    print("SCL verification passed")
