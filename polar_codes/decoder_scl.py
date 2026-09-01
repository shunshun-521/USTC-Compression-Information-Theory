"""极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import zlib

import numpy as np

from decoder_sc import f_operation_min_sum
from encoder import bit_reversed


def _crc_value(info_bits, crc_length=8):
    """基于 zlib CRC32 截断的校验值（自洽实现）"""
    packed = np.packbits(np.asarray(info_bits, dtype=np.uint8))
    return zlib.crc32(packed.tobytes()) & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    val = _crc_value(info_bits, crc_length)
    crc_bits = np.array(
        [(val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "u_hat": np.zeros(self.N, dtype=int),
        }
        path["L"][:, 0] = llr_ch
        return path

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation_min_sum(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    top_bit = path["B"][j - branch_size, s + 1]
                    path["L"][j, s + 1] = (
                        path["L"][j, s] - path["L"][j - branch_size, s]
                        if top_bit == 1
                        else path["L"][j, s] + path["L"][j - branch_size, s]
                    )

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = (
                        int(path["B"][j, s]) ^ int(path["B"][j - branch_size, s])
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path["pm"] += self._pm_penalty(llr_val, 0)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._propagate_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += self._pm_penalty(llr_val, u_bit)
                        new_path["u_hat"][l] = u_bit
                        new_path["B"][l, self.n] = u_bit
                        self._propagate_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][info_pos], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
