"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, _active_bit_level, _active_llr_level, _bit_reversed


# CRC-8: x^8 + x^2 + x + 1 (0x07)
_CRC8_POLY = 0x07
# CRC-16: 0x8005
_CRC16_POLY = 0x8005


def _crc_bits(data, width=8, poly=0x07):
    mask = (1 << width) - 1
    top = 1 << width
    reg = 0
    for bit in data:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ (poly << 1)) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_bits(info_bits.tolist() + [0] * crc_length, crc_length, poly)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_bits(bits.tolist(), crc_length, poly) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        n = self.n
        for path in paths:
            if not path.active:
                continue
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        top_bit = path.B[j - branch_size, s + 1]
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            top_bit,
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        n = self.n
        for path in paths:
            if not path.active:
                continue
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        """与 LLR 符号一致不惩罚，否则加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        br = bit_reversal_permutation(self.N)
        llr_internal = np.asarray(llr_ch, dtype=np.float64)[br]

        paths = [_Path(self.N, self.n, llr_internal.copy())]
        u_results = []

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(paths, l)

            candidates = []
            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr, 0)
                    candidates.append((path.pm + penalty, pidx, 0, path))
                else:
                    for bit in (0, 1):
                        penalty = self._path_metric_penalty(llr, bit)
                        candidates.append((path.pm + penalty, pidx, bit, path))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit, parent in selected:
                child = _Path(self.N, self.n, parent.L[:, 0].copy())
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                child.B[l, self.n] = bit
                new_paths.append(child)

            paths = new_paths
            self._update_bits(paths, l)

        best_path = None
        best_pm = float("inf")
        crc_pass_paths = []

        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass_paths.append((path.pm, u_hat))
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = u_hat

        if crc_pass_paths:
            crc_pass_paths.sort(key=lambda x: x[0])
            return crc_pass_paths[0][1], crc_pass_paths[0][0]

        return best_path, best_pm
