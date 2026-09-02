"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int_,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int_)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    expected = _crc_remainder(payload, poly, crc_length)
    received = 0
    for i, b in enumerate(bits[-crc_length:]):
        received |= int(b) << (crc_length - 1 - i)
    return expected == received


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        if u == hard:
            return pm
        return pm + abs(llr_val)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l, bit_val):
        B[l, self.n] = bit_val
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int_),
                "pm": 0.0,
                "u": np.zeros(N, dtype=np.int_),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_val = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path["pm"], llr_val, 0)
                    child = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                        "u": path["u"].copy(),
                    }
                    child["u"][l] = 0
                    self._update_bits(child["B"], l, 0)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr_val, u)
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                            "u": path["u"].copy(),
                        }
                        child["u"][l] = u
                        self._update_bits(child["B"], l, u)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_ok = [
                crc_check(p["u"][info_idx], self.crc_length) for p in paths
            ]
            if any(crc_ok):
                candidates = [p for p, ok in zip(paths, crc_ok) if ok]
                best = min(candidates, key=lambda p: p["pm"])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best["u"], best["pm"]
