"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import f_boxplus, g_operation, sc_decode_recursive


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(degree):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, deg = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, deg = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, deg)
    crc_bits = np.array([(rem >> (deg - 1 - i)) & 1 for i in range(deg)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, deg = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, deg = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, deg) == 0


def _leaf_llr(llr, frozen_ind, u_known, target_phi, offset=0):
    """已知 u_known（长度 target_phi）时，计算第 target_phi 位的叶子 LLR"""
    n = len(llr)
    if n == 1:
        return float(llr[0])

    half = n // 2
    fi1 = frozen_ind[:half]
    fi2 = frozen_ind[half:]

    llr_u = f_boxplus(llr[:half], llr[half:])

    if target_phi < offset + half:
        u_sub = u_known[: target_phi - offset] if target_phi > offset else np.array([], dtype=int)
        return _leaf_llr(llr_u, fi1, u_sub, target_phi, offset)

    u_left = u_known[:half]
    u_left_up = u_left.astype(float)
    if len(u_left) < half:
        from decoder_sc import _sc_rec_sionna
        _, u_left_up = _sc_rec_sionna(llr_u, fi1.astype(float), True)

    llr_d = g_operation(llr[:half], llr[half:], u_left_up)
    u_sub = u_known[half:target_phi - offset]
    return _leaf_llr(llr_d, fi2, u_sub, target_phi, offset + half)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = self.frozen_bits.astype(float)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _pm_add(self, pm, llr, u):
        llr_c = np.clip(llr, -30, 30)
        return pm + float(np.log1p(np.exp(-(1 - 2 * u) * llr_c)))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"u": np.zeros(self.N, dtype=int), "pm": 0.0}]

        for phi in range(self.N):
            new_paths = []
            for p in paths:
                u_prev = p["u"][:phi]
                llr_leaf = _leaf_llr(llr_ch, self.frozen_ind, u_prev, phi)

                if self.frozen_bits[phi]:
                    opts = [0]
                else:
                    opts = [0, 1]

                for u_bit in opts:
                    np_ = copy.deepcopy(p)
                    np_["u"][phi] = u_bit
                    np_["pm"] = self._pm_add(p["pm"], llr_leaf, u_bit)
                    new_paths.append(np_)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x["pm"])
        best = 0
        if self.crc_length > 0:
            valid = [
                i
                for i, p in enumerate(paths)
                if crc_check(p["u"][self.info_idx], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda i: paths[i]["pm"])

        return paths[best]["u"], paths[best]["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("SCL L=1 vs SC:", np.array_equal(u_sc, u_scl))
