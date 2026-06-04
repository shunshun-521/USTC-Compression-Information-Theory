"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import _bit_reverse_index
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    """模 2 多项式除法求 CRC 余数"""
    reg = [0] * crc_len
    for bit in data_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            poly_bits = [(poly >> i) & 1 for i in range(crc_len - 1, -1, -1)]
            reg = [r ^ p for r, p in zip(reg, poly_bits)]
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(bits[:-crc_length], poly, crc_length)
    return np.array_equal(remainder, bits[-crc_length:])


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组引用）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reverse_index(i, self.n) for i in range(N)]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_penalty(self, llr, u):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs_path(self, path, l):
        L, B = path.L, path.B
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits_path(self, path, l):
        B = path.B
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n) for _ in range(1)]
        paths[0].L[:, 0] = llr_ch.copy()

        for l in self.decode_order:
            for p in paths:
                if p.active:
                    self._update_llrs_path(p, l)

            llr_dec = paths[0].L[l, self.n] if paths else 0.0
            new_paths = []

            if self.frozen_bits[l]:
                for p in paths:
                    if not p.active:
                        continue
                    p.pm += self._path_llr_penalty(p.L[l, self.n], 0)
                    p.B[l, self.n] = 0
                    self._update_bits_path(p, l)
                    new_paths.append(p)
            else:
                for p in paths:
                    if not p.active:
                        continue
                    cur_llr = p.L[l, self.n]
                    for u in (0, 1):
                        cp = _Path(self.N, self.n)
                        cp.L = p.L.copy()
                        cp.B = p.B.copy()
                        cp.pm = p.pm + self._path_llr_penalty(cur_llr, u)
                        cp.B[l, self.n] = u
                        self._update_bits_path(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = float('inf')
        best_path = paths[0]

        for p in paths:
            u_hat = p.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and p.pm < best_pm:
                    best_pm = p.pm
                    best_crc = u_hat
            elif p.pm < best_pm:
                best_pm = p.pm
                best_path = p

        if best_crc is not None:
            return best_crc, best_pm
        return best_path.B[:, self.n].astype(int), best_path.pm
