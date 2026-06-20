"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _decoder_domain_to_natural,
    _frozen_to_decoder_domain,
    _lower_llr,
    _upper_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    """计算 CRC 余数位（MSB first）"""
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    crc_part = _crc_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_part])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    payload = bits[:-crc_length]
    expected = _crc_bits(payload, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _path_metric_penalty(llr, u):
    return 0.0 if u == _llr_to_bit(llr) else abs(llr)


class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("pm", "L", "B", "L_owned", "B_owned")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.L_owned = False
        self.B_owned = False

    def ensure_L(self):
        if not self.L_owned:
            self.L = self.L.copy()
            self.L_owned = True

    def ensure_B(self):
        if not self.B_owned:
            self.B = self.B.copy()
            self.B_owned = True

    def clone(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.L_owned = True
        p.B_owned = True
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = _frozen_to_decoder_domain(frozen_bits, N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        path.ensure_L()
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        path.ensure_B()
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回自然序 u_hat 与路径度量"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = path.clone()
                    new_path.pm += _path_metric_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                elif self.list_size == 1:
                    new_path = path.clone()
                    u = _llr_to_bit(llr)
                    new_path.pm += _path_metric_penalty(llr, u)
                    new_path.B[l, n] = u
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.clone()
                        new_path.pm += _path_metric_penalty(llr, u)
                        new_path.B[l, n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_any = min(paths, key=lambda p: p.pm)
        best_crc = None

        if self.crc_length > 0:
            for path in sorted(paths, key=lambda p: p.pm):
                u_nat = _decoder_domain_to_natural(path.B[:, n].astype(np.int32), N)
                bits = u_nat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    best_crc = path
                    break
            chosen = best_crc if best_crc is not None else best_any
        else:
            chosen = best_any

        u_hat = _decoder_domain_to_natural(chosen.B[:, n].astype(np.int32), N)
        return u_hat, chosen.pm
