"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _upper_llr_exact,
    _lower_llr_exact,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [
            _bit_reversed_index(i, self.n) for i in range(self.N)
        ]

    def _new_path(self, llr=None):
        path = {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }
        if llr is not None:
            path["L"][:, 0] = llr
        return path

    def _copy_path(self, path):
        return {
            "L": path["L"],
            "B": path["B"],
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "_owned": False,
        }

    def _ensure_owned(self, path):
        if not path.get("_owned", True):
            path["L"] = path["L"].copy()
            path["B"] = path["B"].copy()
            path["_owned"] = True

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = _upper_llr_exact(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    path["L"][j, s + 1] = _lower_llr_exact(
                        path["L"][j, s],
                        path["L"][j - branch_size, s],
                        path["B"][j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
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

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        paths = [self._new_path(llr)]
        paths[0]["_owned"] = True

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._ensure_owned(path)
                self._update_llrs(path, l)
                llr_val = path["L"][l, self.n]
                hard_bit = 0 if llr_val >= 0 else 1

                if l in self.frozen_set:
                    penalty = 0.0 if hard_bit == 0 else abs(llr_val)
                    new_path = self._copy_path(path)
                    self._ensure_owned(new_path)
                    new_path["pm"] += penalty
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        self._ensure_owned(new_path)
                        penalty = 0.0 if u_bit == hard_bit else abs(llr_val)
                        new_path["pm"] += penalty
                        new_path["u_hat"][l] = u_bit
                        new_path["B"][l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]
            for p in paths:
                p["_owned"] = False

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_paths = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_idx], self.crc_length)
            ]
            if crc_paths:
                best = min(crc_paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
