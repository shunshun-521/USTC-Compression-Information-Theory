"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
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
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _bit_reversed(i, n):
        result = 0
        for bit in range(n):
            if i & (1 << bit):
                result |= 1 << (n - 1 - bit)
        return result

    @staticmethod
    def _active_llr_level(i, n):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    @staticmethod
    def _active_bit_level(i, n):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def _update_llr(self, L, B, phase_i):
        n, N = self.n, self.N
        l = self._bit_reversed(phase_i, n)
        for s in range(n - self._active_llr_level(l, n), n):
            bs = 2 ** (s + 1)
            brs = bs // 2
            for j in range(l, N, bs):
                if j % bs < brs:
                    L[j, s + 1] = f_operation(L[j, s], L[j + brs, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - brs, s], L[j, s], B[j - brs, s + 1]
                    )
        return L[l, n]

    def _update_bits(self, B, phase_i, u_val):
        n, N = self.n, self.N
        l = self._bit_reversed(phase_i, n)
        B[l, n] = u_val
        if l >= N // 2:
            for s in range(n, n - self._active_bit_level(l, n), -1):
                bs = 2 ** s
                brs = bs // 2
                for j in range(l, -1, -bs):
                    if j % bs >= brs:
                        B[j - brs, s - 1] = int(B[j, s]) ^ int(B[j - brs, s])
                        B[j, s - 1] = B[j, s]

    def _path_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr = llr_ch[bit_reversal_permutation(N)]

        def new_path():
            L = np.zeros((N, n + 1))
            B = np.zeros((N, n + 1), dtype=int)
            L[:, 0] = llr
            return {"L": L, "B": B, "pm": 0.0, "u": np.zeros(N, dtype=int)}

        paths = [new_path()]

        for phase_i in range(N):
            bit_idx = self._bit_reversed(phase_i, n)
            candidates = []

            for path in paths:
                llr_bit = self._update_llr(path["L"], path["B"], phase_i)

                if bit_idx in self.frozen_set:
                    cp = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._path_penalty(llr_bit, 0),
                        "u": path["u"].copy(),
                    }
                    self._update_bits(cp["B"], phase_i, 0)
                    cp["u"][bit_idx] = 0
                    candidates.append(cp)
                else:
                    for u_cand in (0, 1):
                        cp = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._path_penalty(llr_bit, u_cand),
                            "u": path["u"].copy(),
                        }
                        self._update_bits(cp["B"], phase_i, u_cand)
                        cp["u"][bit_idx] = u_cand
                        candidates.append(cp)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"].astype(int), best["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = 0.15
    for _ in range(50):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u_full)) + rng.normal(0, sigma, N), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL must match SC"

    print("SCL L=1 matches SC: OK")
