"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _channel_llr_for_decoder,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=np.int64)

    def _new_path(self, llr):
        return {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
            "llr0": llr,
        }

    def _init_path(self, path, llr):
        path["L"][:, 0] = llr
        path["pm"] = 0.0
        path["u_hat"].fill(0)
        path["B"].fill(0)

    def _update_llrs_for_phase(self, path, phase_idx):
        l = _bit_reversed_index(phase_idx, self.n)
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top = L[j - branch_size, s]
                    btm = L[j, s]
                    bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, btm, bit)

    def _update_bits_after_decision(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_penalty(self, llr_val, bit):
        if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0):
            return 0.0
        return abs(llr_val)

    def decode(self, llr_ch):
        llr = _channel_llr_for_decoder(llr_ch)
        paths = [self._new_path(llr)]
        self._init_path(paths[0], llr)

        for phase_idx in range(self.N):
            l = _bit_reversed_index(phase_idx, self.n)
            for p in paths:
                self._update_llrs_for_phase(p, phase_idx)

            if self.frozen_bits[l]:
                for p in paths:
                    llv = p["L"][l, self.n]
                    p["u_hat"][l] = 0
                    p["B"][l, self.n] = 0
                    p["pm"] += self._path_penalty(llv, 0)
                    self._update_bits_after_decision(p, l)
                continue

            expanded = []
            for p in paths:
                llv = p["L"][l, self.n]
                for bit in (0, 1):
                    cp = copy.deepcopy(p)
                    cp["pm"] += self._path_penalty(llv, bit)
                    cp["u_hat"][l] = bit
                    cp["B"][l, self.n] = bit
                    self._update_bits_after_decision(cp, l)
                    expanded.append(cp)
            expanded.sort(key=lambda x: x["pm"])
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p["u_hat"][self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        return best["u_hat"].copy(), best["pm"]
