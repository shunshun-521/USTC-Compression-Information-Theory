"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level,
    _frozen_to_set,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    """CRC 移位寄存器处理比特序列。"""
    reg = 0
    mask = 1 << (crc_length - 1)
    full_mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & mask:
                reg = ((reg << 1) ^ poly) & full_mask
            else:
                reg = (reg << 1) & full_mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_process(padded, crc_length, poly)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    return np.array_equal(crc_encode(info, crc_length), bits)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "u_hat": np.zeros(self.N, dtype=int), "L": L, "B": B}

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    np_path = self._copy_path(path)
                    np_path["pm"] += self._pm_penalty(llr, 0)
                    np_path["B"][l, self.n] = 0
                    np_path["u_hat"][l] = 0
                    self._update_bits(np_path, l)
                    new_paths.append(np_path)
                else:
                    for u_bit in (0, 1):
                        np_path = self._copy_path(path)
                        np_path["pm"] += self._pm_penalty(llr, u_bit)
                        np_path["B"][l, self.n] = u_bit
                        np_path["u_hat"][l] = u_bit
                        self._update_bits(np_path, l)
                        new_paths.append(np_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_indices = sorted(set(range(self.N)) - self.frozen_set)
            crc_pass = []
            for p in paths:
                info_bits = p["u_hat"][info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            best = min(crc_pass if crc_pass else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import (
        bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, prepare_channel_llr,
    )
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = prepare_channel_llr(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
    assert mismatches == 0
