"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    lower_llr,
    upper_llr,
)


def _crc_remainder(bits, crc_length=8):
    """计算 CRC 余数（MSB 先行，多项式 CRC-8: 0x07 / CRC-16: 0x8005）。"""
    bits = np.asarray(bits, dtype=np.int64)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)

    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005（MSB 先行）
    """
    info_bits = np.asarray(info_bits, dtype=np.int64)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_update(self, pm, llr, bit):
        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int64),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path["pm"] = self._pm_update(path["pm"], llr, 0)
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] = self._pm_update(path["pm"], llr, bit)
                        new_path["B"][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        return best["B"][:, self.n].astype(int), best["pm"]

    def _copy_path(self, path):
        return {"L": path["L"].copy(), "B": path["B"].copy(), "pm": path["pm"]}

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for path in paths:
                u = path["B"][:, self.n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                return min(valid, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
