"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _compute_llr, _map_channel_llr
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg


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
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = _map_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n, L = self.N, self.n, self.list_size

        llrs = []
        bits = []
        for _ in range(L):
            arr = np.full((n + 1, N), -np.inf, dtype=np.float64)
            arr[n, :] = llr_ch
            llrs.append(arr)
            bits.append(np.full((n + 1, N), -1, dtype=np.int8))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        u_hat_paths = [np.zeros(N, dtype=int) for _ in range(L)]

        for idx in range(N):
            dm = np.zeros(L, dtype=np.float64)
            for path in range(L):
                if pm[path] == np.inf:
                    continue
                llr_val = _compute_llr(0, idx, llrs[path], bits[path])
                if self.frozen_bits[idx]:
                    bits[path][0, idx] = 0
                    u_hat_paths[path][idx] = 0
                    pm[path] += 0.0 if llr_val >= 0 else abs(llr_val)
                else:
                    bit = 0 if llr_val >= 0 else 1
                    bits[path][0, idx] = bit
                    u_hat_paths[path][idx] = bit
                    dm[path] = abs(llr_val)

            if not self.frozen_bits[idx] and L > 1:
                pm_dm = np.concatenate([pm, pm + dm])
                idx_sort = np.argsort(pm_dm)

                idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                idx_min_up = idx_sort[L:][idx_sort[L:] < L]

                for low_i, up_i in zip(idx_min_low, idx_min_up):
                    llrs[up_i] = llrs[low_i].copy()
                    bits[up_i] = bits[low_i].copy()
                    u_hat_paths[up_i] = u_hat_paths[low_i].copy()
                    flipped = 1 - u_hat_paths[low_i][idx]
                    u_hat_paths[up_i][idx] = flipped
                    bits[up_i][0, idx] = flipped
                    pm[up_i] = pm_dm[low_i + L]

        if self.crc_length > 0:
            valid = []
            for path in range(L):
                if pm[path] == np.inf:
                    continue
                info_bits = u_hat_paths[path][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: pm[p]) if valid else int(np.argmin(pm))
        else:
            best = int(np.argmin(pm))

        return u_hat_paths[best], pm[best]


if __name__ == "__main__":
    from decoder_sc import sc_decode
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
