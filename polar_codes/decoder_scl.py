"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _SCDCore, _bit_reversed


_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（基于 Vangala SC 内核）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"core": _SCDCore(self.N, self.frozen_bits), "pm": 0.0, "done": False}]
        paths[0]["core"].set_channel_llr(llr_ch)

        for step, l in enumerate(self._decode_order):
            new_paths = []
            for path in paths:
                core = path["core"]
                if path["done"]:
                    new_paths.append(path)
                    continue
                core._update_llrs(l)
                llr_l = core.L[l, core.n]
                if self.frozen_bits[l]:
                    pm = path["pm"] + self._pm_penalty(llr_l, 0)
                    child = {
                        "core": _SCDCore(self.N, self.frozen_bits),
                        "pm": pm,
                        "done": step == self.N - 1,
                    }
                    child["core"].L = core.L.copy()
                    child["core"].B = core.B.copy()
                    child["core"].B[l, core.n] = 0
                    child["core"]._update_bits(l)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr_l, u_bit)
                        child = {
                            "core": _SCDCore(self.N, self.frozen_bits),
                            "pm": pm,
                            "done": step == self.N - 1,
                        }
                        child["core"].L = core.L.copy()
                        child["core"].B = core.B.copy()
                        child["core"].B[l, core.n] = u_bit
                        child["core"]._update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        best = paths[0]
        for p in paths:
            if p["pm"] < best["pm"]:
                best = p
            if self.crc_length > 0:
                u_hat = p["core"].B[:, self.n].astype(int)
                payload = u_hat[~self.frozen_bits]
                if crc_check(payload, self.crc_length) and (
                    best_crc is None or p["pm"] < best_crc["pm"]
                ):
                    best_crc = p

        chosen = best_crc if best_crc is not None else best
        u_hat = chosen["core"].B[:, self.n].astype(int)
        return u_hat, chosen["pm"]
