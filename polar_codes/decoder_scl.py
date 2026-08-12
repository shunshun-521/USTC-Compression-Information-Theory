"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u_hat": np.zeros(N, dtype=int),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path["L"][j, s + 1] = f_operation(
                                path["L"][j, s], path["L"][j + branch_size, s]
                            )
                        else:
                            top_bit = path["B"][j - branch_size, s + 1]
                            path["L"][j, s + 1] = g_operation(
                                path["L"][j - branch_size, s],
                                path["L"][j, s],
                                top_bit,
                            )

                llr_val = path["L"][l, n]

                if self.frozen_bits[l]:
                    p = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                        "pm": path["pm"] + self._pm_penalty(llr_val, 0),
                    }
                    p["u_hat"][l] = 0
                    p["B"][l, n] = 0
                    self._propagate_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                            "pm": path["pm"] + self._pm_penalty(llr_val, u),
                        }
                        p["u_hat"][l] = u
                        p["B"][l, n] = u
                        self._propagate_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"][self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = paths[0]

        return best["u_hat"].copy(), best["pm"]

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            for j in range(l, -1, -block_size):
                if j % block_size >= block_size // 2:
                    path["B"][j - block_size // 2, s - 1] = (
                        path["B"][j, s] + path["B"][j - block_size // 2, s]
                    ) % 2
                    path["B"][j, s - 1] = path["B"][j, s]
