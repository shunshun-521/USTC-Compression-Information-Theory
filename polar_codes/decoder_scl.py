"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)
from encoder import bit_reversal_permutation


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length <= 0 or len(bits) <= crc_length:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]
        self.br = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n = self.N, self.n

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "pm": 0.0,
            "u": np.zeros(N, dtype=int),
            "active": np.ones(N, dtype=bool),
        }]
        paths[0]["L"][:, 0] = llr_ch
        paths[0]["active"][:] = False

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path["pm"], llr_leaf, 0)
                    child = self._fork_path(path)
                    child["pm"] = pm
                    child["u"][l] = 0
                    child["B"][l, n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr_leaf, u)
                        child = self._fork_path(path)
                        child["pm"] = pm
                        child["u"][l] = u
                        child["B"][l, n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            chosen = min(paths, key=lambda p: p["pm"])

        return chosen["u"].copy(), chosen["pm"]

    def _fork_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u": path["u"].copy(),
            "active": path["active"].copy(),
        }

    def _update_llrs(self, path, l):
        if path["active"][l]:
            return
        start_s = self.n - _active_llr_level(l, self.n)
        L = path["L"]
        B = path["B"]
        N = self.N
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)
        path["active"][l] = True

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        start_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, start_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
