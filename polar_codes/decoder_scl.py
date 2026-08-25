"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_scalar,
    _lower_llr,
    _update_bits,
    _upper_llr,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    """单条 SCL 路径（Lazy Copy）。"""
    __slots__ = ("L", "B", "pm", "parent", "branch_id")

    def __init__(self, N, n, parent=None, branch_id=0):
        self.L = None
        self.B = None
        self.pm = 0.0
        self.parent = parent
        self.branch_id = branch_id
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan)
        else:
            self.L = parent.L
            self.B = parent.B

    def copy_state(self, N, n):
        self.L = self.L.copy()
        self.B = self.B.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _init_paths(self, llr_ch):
        path = _Path(self.N, self.n)
        for i in range(self.N):
            path.L[i, 0] = llr_ch[self.br[i]]
        return [path]

    def _update_llrs_path(self, path, l):
        L, B = path.L, path.B
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = _upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits_path(self, path, l):
        _update_bits(path.B, l, self.n, self.N)

    def _pm_penalty(self, llr_val, u_bit):
        """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat（最优路径），pm（路径度量）
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = _bit_reversed_scalar(i, self.n)
            new_paths = []

            for path in paths:
                path.copy_state(self.N, self.n)
                self._update_llrs_path(path, l)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_leaf, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    self._update_bits_path(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.n, parent=path, branch_id=u_bit)
                        child.copy_state(self.N, self.n)
                        child.pm = path.pm + self._pm_penalty(llr_leaf, u_bit)
                        child.B[l, self.n] = u_bit
                        self._update_bits_path(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        # 选择最优路径
        crc_pass = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_idx = np.where(self.frozen_bits == 0)[0]
                info_bits = u_hat[info_idx]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        if crc_pass:
            best = min(crc_pass, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
