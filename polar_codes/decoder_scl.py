"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    n = (len(bits) + 7) // 8
    out = bytearray()
    for i in range(n):
        byte = 0
        for j in range(8):
            idx = i * 8 + j
            byte = (byte << 1) | (int(bits[idx]) if idx < len(bits) else 0)
        out.append(byte)
    return bytes(out)


def _crc8_byte(data, poly=CRC8_POLY):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(data, poly=CRC16_POLY):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        crc_val = _crc8_byte(_bits_to_bytes(info_bits))
        crc_bits = np.array([(crc_val >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        crc_val = _crc16_byte(_bits_to_bytes(info_bits))
        crc_bits = np.array([(crc_val >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    if crc_length == 8:
        return _crc8_byte(_bits_to_bytes(bits)) == 0
    return _crc16_byte(_bits_to_bytes(bits)) == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（置换 SC 结构）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.br = bit_reversal_permutation(N)
        self.frozen_internal = self.frozen_bits[self.br]
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [_Path(N, n) for _ in range(L)]
        paths[0].L[:, 0] = llr_ch
        active_count = 1

        for l in self.decode_order:
            candidates = []
            for pidx in range(active_count):
                path = paths[pidx]
                self._update_llrs(path, l)
                llr0 = path.L[l, n]
                if self.frozen_internal[l]:
                    candidates.append((path.pm + _pm_penalty(llr0, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((path.pm + _pm_penalty(llr0, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]

            new_paths = [_Path(N, n) for _ in range(L)]
            for i, (pm, src, u_bit) in enumerate(candidates):
                src_path = paths[src]
                new_paths[i].L[:] = src_path.L
                new_paths[i].B[:] = src_path.B
                new_paths[i].pm = pm
                new_paths[i].B[l, n] = u_bit
                self._update_bits(new_paths[i], l)

            paths = new_paths
            active_count = len(candidates)

        best_idx = 0
        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            passed = []
            for i in range(active_count):
                u_nat = paths[i].B[:, n].astype(int)[self.br]
                if crc_check(u_nat[info_pos], self.crc_length):
                    passed.append(i)
            if passed:
                best_idx = min(passed, key=lambda i: paths[i].pm)
            else:
                best_idx = min(range(active_count), key=lambda i: paths[i].pm)
        else:
            best_idx = min(range(active_count), key=lambda i: paths[i].pm)

        u_internal = paths[best_idx].B[:, n].astype(int)
        return u_internal[self.br], paths[best_idx].pm
