"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB-first）"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int64)])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int64)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _path_metric_penalty(llr, u_bit):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u_bit == preferred else abs(llr)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表规模适中时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._rev = np.array([_bit_reversed(i, self.n) for i in range(N)], dtype=np.int64)

    def _init_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int64)
        L[:, 0] = llr_ch[self._rev]
        return {"L": L, "B": B, "pm": 0.0}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._init_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + _path_metric_penalty(llr, 0),
                    }
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _path_metric_penalty(llr, bit),
                        }
                        new_path["B"][l, self.n] = bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        best_u = None
        for path in paths:
            u_hat = path["B"][:, self.n].astype(int)
            pm = path["pm"]
            if self.crc_length > 0:
                info_mask = ~self.frozen_bits
                payload = u_hat[info_mask]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or pm < best_crc:
                        best_crc = pm
                        best_u = u_hat
            if best_pm is None or pm < best_pm:
                best_pm = pm
                if best_crc is None:
                    best_u = u_hat

        if best_crc is not None:
            return best_u, best_crc
        return best_u, best_pm
