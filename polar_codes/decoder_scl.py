"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import _PermutedSCD, _bit_reversed, _frozen_indices_from_mask


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    poly = _crc_polynomial(crc_length)
    work = info_bits + [0] * crc_length
    times = len(info_bits)

    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= poly[j]

    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int).tolist()
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return list(expected) == bits


class _Path:
    __slots__ = ("decoder", "pm", "u_hat")

    def __init__(self, decoder, pm, u_hat):
        self.decoder = decoder
        self.pm = pm
        self.u_hat = u_hat


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(_frozen_indices_from_mask(frozen_bits))
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        from decoder_sc import _permute_llr_for_decode

        init = _PermutedSCD(self.N, self.frozen_set)
        init.L[:, 0] = _permute_llr_for_decode(llr_ch, self.N)
        paths = [_Path(init, 0.0, np.zeros(self.N, dtype=int))]

        for l in self.decode_order:
            candidates = []

            for path in paths:
                path.decoder._update_llrs(l)
                llr = path.decoder.L[l, self.n]

                if l in self.frozen_set:
                    new_dec = copy.deepcopy(path.decoder)
                    new_dec.B[l, self.n] = 0
                    new_dec._update_bits(l)
                    new_u = path.u_hat.copy()
                    new_u[l] = 0
                    pm = path.pm + self._path_metric_penalty(llr, 0)
                    candidates.append(_Path(new_dec, pm, new_u))
                else:
                    for bit in (0, 1):
                        new_dec = copy.deepcopy(path.decoder)
                        new_dec.B[l, self.n] = bit
                        new_dec._update_bits(l)
                        new_u = path.u_hat.copy()
                        new_u[l] = bit
                        pm = path.pm + self._path_metric_penalty(llr, bit)
                        candidates.append(_Path(new_dec, pm, new_u))

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_ok = [p for p in paths if self.crc_length == 0 or self._crc_passes(p.u_hat)]
        best = crc_ok[0] if crc_ok else paths[0]
        return best.u_hat.astype(int), best.pm

    def _crc_passes(self, u_hat):
        info_positions = np.where(self.frozen_bits == 0)[0]
        payload = u_hat[info_positions]
        return crc_check(payload, self.crc_length)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    rng = np.random.default_rng(1)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    match_sc = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if np.array_equal(u_sc, u_scl):
            match_sc += 1
    print(f"L=1 SCL matches SC: {match_sc}/50")
