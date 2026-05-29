"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_utils import (
    upper_llr,
    lower_llr,
    hard_decision,
    active_llr_level,
    active_bit_level,
    bit_reversed,
)

_CRC8_FUN = None
_CRC16_FUN = None


def _get_crc_fun(length):
    global _CRC8_FUN, _CRC16_FUN
    if length == 8:
        if _CRC8_FUN is None:
            import crcmod
            _CRC8_FUN = crcmod.mkCrcFun(0x107, initCrc=0, xorOut=0)
        return _CRC8_FUN, 8
    if length == 16:
        if _CRC16_FUN is None:
            import crcmod
            _CRC16_FUN = crcmod.mkCrcFun(0x11021, initCrc=0, xorOut=0)
        return _CRC16_FUN, 16
    raise ValueError("crc_length must be 8 or 16")


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int).tolist()
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC-8/16 并附加到信息比特后（多项式 0x07 / 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_fun, width = _get_crc_fun(crc_length)
    crc_val = crc_fun(_bits_to_bytes(info_bits))
    crc_bits = np.array([(crc_val >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    _, width = _get_crc_fun(crc_length)
    if len(bits) < width:
        return False
    payload = bits[:-width]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(int(i) for i in np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = self.L_size

        paths = [{
            'pm': 0.0,
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phase in [bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                self._update_llrs_path(path, phase)
                llr_dec = path['L'][phase, n]
                if phase in self.frozen_set:
                    u = 0
                    pm = self._pm_update(path['pm'], llr_dec, u)
                    new_paths.append(self._fork_path(path, phase, u, pm))
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path['pm'], llr_dec, u)
                        new_paths.append(self._fork_path(path, phase, u, pm))

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]

        best = self._select_path(paths)
        return best['u'].copy(), best['pm']

    def _fork_path(self, path, phase, u, pm):
        child = {
            'pm': pm,
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u': path['u'].copy(),
        }
        child['B'][phase, self.n] = u
        child['u'][phase] = u
        if phase >= self.N // 2:
            self._update_bits_path(child, phase)
        return child

    def _update_llrs_path(self, path, phase):
        Larr, Barr = path['L'], path['B']
        for s in range(self.n - active_llr_level(phase, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(phase, self.N, block_size):
                if j % block_size < branch_size:
                    Larr[j, s + 1] = upper_llr(Larr[j, s], Larr[j + branch_size, s])
                else:
                    top_bit = Barr[j - branch_size, s + 1]
                    Larr[j, s + 1] = lower_llr(
                        Larr[j, s], Larr[j - branch_size, s], top_bit
                    )

    def _update_bits_path(self, path, phase):
        Barr = path['B']
        for s in range(self.n, self.n - active_bit_level(phase, self.n), -1):
            block_size = int(2 ** s)
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    Barr[j - branch_size, s - 1] = int(Barr[j, s]) ^ int(
                        Barr[j - branch_size, s]
                    )
                    Barr[j, s - 1] = Barr[j, s]

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_part = p['u'][self.info_indices]
                if crc_check(info_part, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p['pm'])
        return min(paths, key=lambda p: p['pm'])
