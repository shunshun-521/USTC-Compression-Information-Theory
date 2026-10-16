"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _prepare_llr, _polar_decode_sc

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr, u):
    if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
        return pm
    return pm + abs(llr)


def _scl_recurse(paths, llr_blk, frozen_blk, list_size):
    n = len(llr_blk)
    frozen_blk = np.asarray(frozen_blk, dtype=int)

    if n == 1:
        new_paths = []
        for p in paths:
            llr0 = llr_blk[0]
            if frozen_blk[0]:
                new_paths.append(
                    {
                        "pm": _path_metric_update(p["pm"], llr0, 0),
                        "u": np.array([0], dtype=int),
                        "u_up": np.array([0.0]),
                    }
                )
            else:
                for bit in (0, 1):
                    new_paths.append(
                        {
                            "pm": _path_metric_update(p["pm"], llr0, bit),
                            "u": np.array([bit], dtype=int),
                            "u_up": np.array([float(bit)]),
                        }
                    )
        new_paths.sort(key=lambda x: x["pm"])
        return new_paths[:list_size]

    half = n // 2
    llr1 = llr_blk[:half]
    llr2 = llr_blk[half:]
    frozen1 = frozen_blk[:half]
    frozen2 = frozen_blk[half:]

    upper_llr = f_operation(llr1, llr2)
    paths_upper = _scl_recurse(paths, upper_llr, frozen1, list_size)

    expanded = []
    for p in paths_upper:
        lower_llr = g_operation(llr1, llr2, p["u_up"])
        sub = _scl_recurse([{"pm": p["pm"], "u": None, "u_up": None}], lower_llr, frozen2, list_size)
        for sp in sub:
            u_cat = np.concatenate([p["u"], sp["u"]])
            u_up_cat = np.concatenate(
                [
                    (p["u_up"].astype(np.int8) ^ sp["u_up"].astype(np.int8)).astype(
                        np.float64
                    ),
                    sp["u_up"],
                ]
            )
            expanded.append({"pm": sp["pm"], "u": u_cat, "u_up": u_up_cat})

    expanded.sort(key=lambda x: x["pm"])
    return expanded[:list_size]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        paths = _scl_recurse(
            [{"pm": 0.0, "u": np.array([], dtype=int), "u_up": np.array([], dtype=np.float64)}],
            llr,
            self.frozen_bits,
            self.list_size,
        )

        if self.crc_length > 0:
            for p in paths:
                info = p["u"][self.info_indices]
                if crc_check(info, self.crc_length):
                    return p["u"].copy(), p["pm"]

        best = paths[0]
        return best["u"].copy(), best["pm"]
