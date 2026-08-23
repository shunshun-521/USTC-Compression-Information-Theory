"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    precompute_sc_indices,
    sc_decode,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_remainder(bits, generator):
    msg = [int(b) for b in bits] + [0] * (len(generator) - 1)
    for i in range(len(bits)):
        if msg[i]:
            for j in range(len(generator)):
                msg[i + j] ^= generator[j]
    return msg[len(bits) :]


def _crc_verify(bits, generator):
    msg = [int(b) for b in bits]
    for i in range(len(bits) - len(generator) + 1):
        if msg[i]:
            for j in range(len(generator)):
                msg[i + j] ^= generator[j]
    return all(x == 0 for x in msg[-(len(generator) - 1) :])


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    generator = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, generator)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    generator = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_verify(bits, generator)


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n, self.decode_order = precompute_sc_indices(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = Path(self.n, self.N)
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat
        return dst

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        paths = [Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[l]:
                    p = self._copy_path(path)
                    p.pm += self._pm_penalty(llr0, 0)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr0, bit)
                        p.u_hat[l] = bit
                        p.B[l, self.n] = bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(5.0, K / N)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("L=1 matches SC:", np.array_equal(u_sc, u_scl))
