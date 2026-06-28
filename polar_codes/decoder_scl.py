"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
    _align_channel_llr,
    _prepare_frozen,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    mask = 1 << (crc_length - 1)
    full_mask = (1 << crc_length) - 1
    if reg & mask:
        reg = ((reg << 1) ^ poly) & full_mask
    else:
        reg = (reg << 1) & full_mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    for _ in range(crc_length):
        reg = _crc_step(reg, 0, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _prepare_frozen(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_update(self, pm, llr, u):
        penalty = 0.0 if u == (0 if llr >= 0 else 1) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = _align_channel_llr(llr_ch, self.N)

        paths = [{
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                self._update_llrs(L, B, l)
                llr = L[l, self.n]

                if l in self.frozen_set:
                    pm = self._path_metric_update(path["pm"], llr, 0)
                    B[l, self.n] = 0
                    u_hat = path["u_hat"].copy()
                    u_hat[l] = 0
                    self._update_bits(B, l)
                    candidates.append({
                        "L": L, "B": B, "pm": pm, "u_hat": u_hat,
                    })
                else:
                    for u in (0, 1):
                        Bc = B.copy()
                        u_hat = path["u_hat"].copy()
                        Bc[l, self.n] = u
                        u_hat[l] = u
                        self._update_bits(Bc, l)
                        candidates.append({
                            "L": L.copy(),
                            "B": Bc,
                            "pm": self._path_metric_update(path["pm"], llr, u),
                            "u_hat": u_hat,
                        })

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p["u_hat"])]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _crc_valid(self, u_hat):
        info_idx = np.where(~self.frozen_bits)[0]
        payload = u_hat[info_idx]
        if len(payload) < self.crc_length:
            return False
        return crc_check(payload, self.crc_length)
