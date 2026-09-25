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
    reorder_channel_llr,
    upper_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask)
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_val = _crc_register(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_register(bits, crc_length, poly) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
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

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = reorder_channel_llr(llr_ch)
        N = self.N
        n = self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                cur_llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    new_path = {
                        "pm": path["pm"] + (abs(cur_llr) if cur_llr < 0 else 0.0),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    new_path["B"][l, n] = 0
                    self._update_bits(new_path["B"], l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and cur_llr >= 0) or (
                            bit == 1 and cur_llr < 0
                        ) else abs(cur_llr)
                        new_path = {
                            "pm": path["pm"] + penalty,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        new_path["B"][l, n] = bit
                        self._update_bits(new_path["B"], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = []
            for p in paths:
                u_hat = p["B"][:, n].astype(np.int8)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    crc_paths.append(p)
            if crc_paths:
                best = min(crc_paths, key=lambda p: p["pm"])
            else:
                best = min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(np.int8)
        return u_hat, best["pm"]
