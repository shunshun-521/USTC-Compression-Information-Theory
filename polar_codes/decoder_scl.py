"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB 优先）。"""
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = bits[-crc_length:]
    computed = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(expected, computed)


class PathState:
    """单条 SCL 路径状态（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)

    def _copy_path(self, src, dst):
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.active = True

    def _path_llr(self, path, l):
        """计算路径 path 在索引 l 处的 LLR。"""
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, self.n]

    def _path_update_bits(self, path, l):
        B = path.B
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr_val, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_aligned = np.asarray(llr_ch, dtype=np.float64)[self.brp]

        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_aligned

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                llr_val = self._path_llr(path, l)

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr_val, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._path_update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._pm_penalty(llr_val, bit)
                        candidates.append((new_pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit in selected:
                if len(new_paths) >= self.list_size:
                    break
                src = paths[pidx]
                if bit is None:
                    dst = PathState(self.N, self.n)
                    self._copy_path(src, dst)
                    new_paths.append(dst)
                else:
                    dst = PathState(self.N, self.n)
                    self._copy_path(src, dst)
                    dst.pm = pm
                    dst.B[l, self.n] = bit
                    dst.u_hat[l] = bit
                    self._path_update_bits(dst, l)
                    new_paths.append(dst)

            paths = new_paths

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
