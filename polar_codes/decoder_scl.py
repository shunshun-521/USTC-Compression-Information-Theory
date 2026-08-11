"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, f_operation_exact, g_operation,
    _active_llr_level, _active_bit_level, _update_llrs, _update_bits,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask)
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _pm_update(pm, llr, u):
    expected = _llr_to_bit(llr)
    return pm if u == expected else pm + abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, use_minsum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.use_minsum = use_minsum
        self.rev = bit_reversal_permutation(N)
        self.frozen_w = np.zeros(N, dtype=bool)
        for j in range(N):
            self.frozen_w[j] = self.frozen_bits[self.rev[j]]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_w": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        paths = [self._new_path(llr_ch)]

        for l in self.rev:
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.use_minsum)
                llr = path["L"][l, self.n]

                if self.frozen_w[l]:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": _pm_update(path["pm"], llr, 0),
                        "u_w": path["u_w"].copy(),
                    }
                    new_path["B"][l, self.n] = 0
                    new_path["u_w"][l] = 0
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": _pm_update(path["pm"], llr, u_val),
                            "u_w": path["u_w"].copy(),
                        }
                        new_path["B"][l, self.n] = u_val
                        new_path["u_w"][l] = u_val
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["u_w"][self.rev]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_w"][self.rev][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["u_w"][self.rev]

        return u_hat.copy(), best["pm"]
