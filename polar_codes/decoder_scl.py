"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                break
    if crc_length == 16:
        reg = 0
        for b in bits:
            reg ^= int(b) << 15
            for _ in range(1):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = CRC8_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 7
            for _ in range(8):
                if rem & 0x80:
                    rem = ((rem << 1) ^ poly) & 0xFF
                else:
                    rem = (rem << 1) & 0xFF
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = CRC16_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 15
            for _ in range(16):
                if rem & 0x8000:
                    rem = ((rem << 1) ^ poly) & 0xFFFF
                else:
                    rem = (rem << 1) & 0xFFFF
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC；bits 含信息位 + CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = CRC8_POLY
        rem = 0
        for b in bits:
            rem ^= int(b) << 7
            for _ in range(8):
                if rem & 0x80:
                    rem = ((rem << 1) ^ poly) & 0xFF
                else:
                    rem = (rem << 1) & 0xFF
        return rem == 0
    if crc_length == 16:
        poly = CRC16_POLY
        rem = 0
        for b in bits:
            rem ^= int(b) << 15
            for _ in range(16):
                if rem & 0x8000:
                    rem = ((rem << 1) ^ poly) & 0xFFFF
                else:
                    rem = (rem << 1) & 0xFFFF
        return rem == 0
    raise ValueError("crc_length must be 8 or 16")


class Path:
    """单条 SCL 路径（Lazy Copy：共享 L/B 数组，仅在写入时复制层切片）"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = Path.__new__(Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（polarcodes 风格 LLR 更新 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        stop = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, stop, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self.rev[i]
            candidates = []

            for path in paths:
                self._update_llrs(path.L, path.B, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    new_pm = path.pm + self._pm_penalty(llr, 0)
                    candidates.append((new_pm, path, bit))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._pm_penalty(llr, bit)
                        candidates.append((new_pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = parent.copy()
                child.pm = pm
                child.B[l, self.n] = bit
                child.u_hat[l] = bit
                self._update_bits(child.B, l)
                new_paths.append(child)
            paths = new_paths

        best_pm = float("inf")
        best_u = paths[0].u_hat
        crc_ok_paths = []

        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok_paths.append(path)
            if path.pm < best_pm:
                best_pm = path.pm
                best_u = path.u_hat

        if self.crc_length > 0 and crc_ok_paths:
            best = min(crc_ok_paths, key=lambda p: p.pm)
            return best.u_hat.copy(), best.pm
        return best_u.copy(), best_pm
