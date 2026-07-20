"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _li


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
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(extended, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _init_path(self, llr_ch):
        llrs = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
        llrs[self.n, :] = llr_ch.copy()
        s = -np.ones((self.n + 1, self.N), dtype=np.int32)
        return llrs, s

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        llrs, s = self._init_path(llr_ch)
        llrs_list = [llrs]
        s_list = [s]
        pm = [0.0]

        for phi in range(self.N):
            candidates_llrs = []
            candidates_s = []
            candidates_pm = []

            for pidx in range(len(llrs_list)):
                llrs = llrs_list[pidx]
                s = s_list[pidx]
                cur_pm = pm[pidx]
                llrs[0, phi] = _li(0, phi, llrs, s)
                llr_val = llrs[0, phi]

                if self.frozen_bits[phi]:
                    candidates_llrs.append(llrs)
                    candidates_s.append(s)
                    candidates_pm.append(cur_pm + (-llr_val * (llr_val < 0)))
                    s[0, phi] = 0
                else:
                    hard = 0 if llr_val >= 0 else 1
                    for bit in (0, 1):
                        llrs_c = llrs.copy()
                        s_c = s.copy()
                        s_c[0, phi] = bit
                        penalty = 0.0 if bit == hard else abs(llr_val)
                        candidates_llrs.append(llrs_c)
                        candidates_s.append(s_c)
                        candidates_pm.append(cur_pm + penalty)

            order = np.argsort(candidates_pm)[:L]
            llrs_list = [candidates_llrs[i] for i in order]
            s_list = [candidates_s[i] for i in order]
            pm = [candidates_pm[i] for i in order]

        best = int(np.argmin(pm))
        if self.crc_length > 0:
            valid = [i for i, s in enumerate(s_list)
                     if crc_check(s[0, self.info_indices], self.crc_length)]
            if valid:
                best = valid[int(np.argmin([pm[i] for i in valid]))]

        return s_list[best][0, :].astype(int), float(pm[best])


def verify_scl_equals_sc():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("SCL L=1 verification passed")


if __name__ == "__main__":
    verify_scl_equals_sc()
