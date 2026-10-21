"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
from decoder_sc import sc_decode_recursive, _f_scalar, _g_scalar


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in np.asarray(info_bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


def _llr_at_phi(llr_ch, frozen, prior_bits, phi):
    """在已有 prior_bits[0..phi-1] 时计算比特 phi 的 LLR（与 SC 树一致）"""
    llr = list(np.asarray(llr_ch, dtype=np.float64))
    frozen = np.asarray(frozen, dtype=bool)
    decisions = {}

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if i == phi:
                return llrs[0]
            if i in prior_bits:
                decisions[i] = prior_bits[i]
            elif frozen[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [_f_scalar(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = _sc(left_llr, indices[:half])
        if isinstance(enc_left, float):
            return enc_left
        right_llr = [
            _g_scalar(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, indices[half:])
        if isinstance(enc_right, float):
            return enc_right
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    return float(_sc(llr, list(range(len(llr_ch)))))


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_positions = np.where(~self.frozen)[0]

    @staticmethod
    def _penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen)
            return u, 0.0

        paths = [{"pm": 0.0, "bits": {}}]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                llr = _llr_at_phi(llr_ch, self.frozen, path["bits"], phi)
                if self.frozen[phi]:
                    path["pm"] += self._penalty(llr, 0)
                    path["bits"][phi] = 0
                    expanded.append(path)
                else:
                    for u in (0, 1):
                        nb = dict(path["bits"])
                        nb[phi] = u
                        expanded.append(
                            {
                                "pm": path["pm"] + self._penalty(llr, u),
                                "bits": nb,
                            }
                        )
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.L]

        def to_u(path):
            u = np.zeros(self.N, dtype=int)
            for k, v in path["bits"].items():
                u[k] = v
            return u

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(to_u(p)[self.info_positions], self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return to_u(best), best["pm"]


def scl_equivalent_sc(llr_ch, frozen_bits):
    u_scl, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1).decode(llr_ch)
    u_sc = sc_decode_recursive(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
