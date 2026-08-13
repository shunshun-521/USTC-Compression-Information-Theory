"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = _crc_remainder(bits[:-crc_length], crc_length)
    received = 0
    for bit in bits[-crc_length:]:
        received = (received << 1) | int(bit)
    return expected == received


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _scl(self, y, depth, node, paths):
        if depth == self.n - 1:
            out = []
            for pm, u in paths:
                if node in self.frozen_set:
                    u = u.copy()
                    u[node] = 0
                    out.append((pm + self._penalty(y[0], 0), u))
                else:
                    for bit in (0, 1):
                        u_copy = u.copy()
                        u_copy[node] = bit
                        out.append((pm + self._penalty(y[0], bit), u_copy))
            out.sort(key=lambda item: item[0])
            return out[: self.list_size]

        half = len(y) // 2
        left_y = y[:half]
        right_y = y[half:]
        left_paths = self._scl(f_operation(left_y, right_y), depth + 1, 2 * node, paths)

        merged = []
        for pm, u in left_paths:
            left_bits = []
            self._gather_bits(u, 2 * node, half, left_bits)
            right_llr = g_operation(left_y, right_y, left_bits)
            merged.extend(self._scl(right_llr, depth + 1, 2 * node + 1, [(pm, u)]))

        merged.sort(key=lambda item: item[0])
        return merged[: self.list_size]

    def _gather_bits(self, u, node, span, out):
        if span == 1:
            out.append(u[node])
            return
        h = span // 2
        self._gather_bits(u, node, h, out)
        self._gather_bits(u, node + h, h, out)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl(llr_ch, 0, 0, [(0.0, np.zeros(self.N, dtype=int))])

        if self.crc_length > 0:
            valid = [
                (pm, u) for pm, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pm, u_hat = min(valid if valid else paths, key=lambda item: item[0])
        else:
            pm, u_hat = min(paths, key=lambda item: item[0])
        return u_hat.copy(), pm


def _run_scl_self_tests():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(20.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_src)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL should match SC"

    assert crc_check(crc_encode(np.array([1, 0, 1, 1]), 8), 8)


if __name__ == "__main__":
    _run_scl_self_tests()
    print("SCL decoder self-tests passed.")
