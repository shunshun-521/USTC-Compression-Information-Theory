"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），与 Permuted SCD 结构一致
"""
import copy
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(info_bits, dtype=int), np.asarray(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Permuted SC 列表扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_nat = np.asarray(frozen_bits, dtype=int) == 1
        self.frozen_idx = set(np.where(self.frozen_nat)[0])

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [
            {
                "L": np.full((N, n + 1), np.nan),
                "B": np.full((N, n + 1), np.nan),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for step, l in enumerate([_bit_reversed(i, n) for i in range(N)]):
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path["L"][l, n]
                if l in self.frozen_idx:
                    pm = _pm_update(path["pm"], cur_llr, 0)
                    np_path = copy.deepcopy(path)
                    np_path["pm"] = pm
                    np_path["u_hat"][l] = 0
                    np_path["B"][l, n] = 0
                    self._update_bits(np_path, l)
                    new_paths.append(np_path)
                else:
                    for u in (0, 1):
                        pm = _pm_update(path["pm"], cur_llr, u)
                        np_path = copy.deepcopy(path)
                        np_path["pm"] = pm
                        np_path["u_hat"][l] = u
                        np_path["B"][l, n] = u
                        self._update_bits(np_path, l)
                        new_paths.append(np_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_nat = p["u_hat"][~self.frozen_nat]
                if crc_check(info_nat, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        L = path["L"]
        B = path["B"]
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        B = path["B"]
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
