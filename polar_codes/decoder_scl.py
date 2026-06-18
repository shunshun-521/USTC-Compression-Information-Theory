"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int):
        reg ^= bit << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    返回 True/False。
    """
    return np.array_equal(
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
        bits[-crc_length:],
    )


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 P/C 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = (
            precompute_sc_indices(N)
        )
        self.llr_mem = 2 * N

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _update_llr(self, P, C, phi):
        n = self.n
        for layer in self.llr_layer_vec[phi]:
            lam_next = self.lambda_offset[layer + 1]
            half = 1 << (n - 1 - layer)
            for i in range(0, 2 ** layer):
                base = i * (half * 2)
                left = lam_next + base
                right = left + half
                P[layer, left : left + half] = f_operation(
                    P[layer + 1, left : left + half],
                    P[layer + 1, right : right + half],
                )
                P[layer, right : right + half] = g_operation(
                    P[layer + 1, left : left + half],
                    P[layer + 1, right : right + half],
                    C[layer, i],
                )

    def _update_bits(self, C, phi, u_bit):
        n = self.n
        for layer in self.bit_layer_vec[phi]:
            half = 1 << (n - 1 - layer)
            for i in range(0, 2 ** layer):
                if phi % 2 == 0:
                    C[layer + 1, 2 * i] = (C[layer, i] + u_bit) % 2
                    C[layer + 1, 2 * i + 1] = u_bit
                else:
                    C[layer + 1, 2 * i + 1] = u_bit

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [
            {
                "pm": 0.0,
                "P": np.zeros((n + 1, self.llr_mem)),
                "C": np.zeros((n + 1, N), dtype=int),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["P"][n, self.lambda_offset[n] : self.lambda_offset[n] + N] = llr_ch

        for phi in range(N):
            candidates = []
            for path in paths:
                self._update_llr(path["P"], path["C"], phi)
                llr = path["P"][0, 0]
                if self.frozen_bits[phi]:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, u_bit in candidates[: self.list_size]:
                child = {
                    "pm": pm,
                    "P": parent["P"].copy(),
                    "C": parent["C"].copy(),
                    "u": parent["u"].copy(),
                }
                child["u"][phi] = u_bit
                self._update_bits(child["C"], phi, u_bit)
                new_paths.append(child)
            paths = new_paths

        best_pm = float("inf")
        best_u = None
        crc_pm = float("inf")
        crc_u = None

        for path in paths:
            u_hat = path["u"]
            if self.crc_length > 0:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length) and path["pm"] < crc_pm:
                    crc_pm = path["pm"]
                    crc_u = u_hat.copy()
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = u_hat.copy()

        if crc_u is not None:
            return crc_u, crc_pm
        return best_u, best_pm


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(12.0, K / N)

    mismatches = 0
    for _ in range(50):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC mismatches: {mismatches}/50")
    assert mismatches == 0

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits[:4], 8)
    assert crc_check(coded, 8)
    print("SCL/CRC self-tests passed.")
