"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


def _pm_add(pm, llr, u):
    if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
        return pm
    return pm + abs(llr)


def _scl_recursive(llr, frozen, L, pm_in=0.0):
    """递归 SCL，返回路径列表：dict(pm, u_hat_slice, u_up_slice)。"""
    n = len(llr)
    if n == 1:
        paths = []
        if frozen[0]:
            paths.append({"pm": _pm_add(pm_in, llr[0], 0), "u": np.array([0], dtype=int), "up": np.array([0], dtype=int)})
        else:
            for u in (0, 1):
                paths.append(
                    {
                        "pm": _pm_add(pm_in, llr[0], u),
                        "u": np.array([u], dtype=int),
                        "up": np.array([u], dtype=int),
                    }
                )
        return paths

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    f1, f2 = frozen[:half], frozen[half:]

    left_paths = _scl_recursive(f_operation(llr1, llr2), f1, L, pm_in)
    all_paths = []
    for lp in left_paths:
        llr_r = g_operation(llr1, llr2, lp["up"])
        right_paths = _scl_recursive(llr_r, f2, L, lp["pm"])
        for rp in right_paths:
            u = np.concatenate([lp["u"], rp["u"]])
            up_left = (lp["up"] ^ rp["up"]).astype(int)
            up = np.concatenate([up_left, rp["up"]])
            all_paths.append({"pm": rp["pm"], "u": u, "up": up})

    all_paths.sort(key=lambda p: p["pm"])
    return all_paths[:L]


class SCLDecoder:
    """SCL / CA-SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = _scl_recursive(llr_ch, self.frozen_bits, self.L)
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            good = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            best = min(good if good else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u"].astype(int), float(best["pm"])
