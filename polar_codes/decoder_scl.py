"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import active_bit_level, active_llr_level, f_operation, g_operation


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch=None):
        path = {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }
        if llr_ch is not None:
            path["L"][:, 0] = llr_ch
        return path

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    top_bit = path["B"][j - branch_size, s + 1]
                    path["L"][j, s + 1] = g_operation(
                        path["L"][j - branch_size, s], path["L"][j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = (
                        path["B"][j, s] ^ path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr, 0)
                    new_path = path
                    new_path = copy.copy(path)
                    new_path["L"] = path["L"]
                    new_path["B"] = path["B"]
                    new_path["u_hat"] = path["u_hat"].copy()
                    new_path["pm"] = path["pm"] + penalty
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append((new_path["pm"], new_path))
                else:
                    for bit in (0, 1):
                        new_path = {
                            "L": path["L"],
                            "B": path["B"],
                            "pm": path["pm"] + self._pm_penalty(llr, bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"] = path["B"].copy()
                        new_path["B"][l, self.n] = bit
                        new_path["u_hat"][l] = bit
                        self._update_bits(new_path, l)
                        candidates.append((new_path["pm"], new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(8.0, K / N)
    scl1 = SCLDecoder(N, frozen_bits, list_size=1)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma
        )
        uh_sc = sc_decode(llr, frozen_bits)
        uh_scl, _ = scl1.decode(llr)
        if not np.array_equal(uh_sc, uh_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
    assert mismatches == 0, "SCL L=1 should match SC"
    print("SCL decoder tests passed.")
