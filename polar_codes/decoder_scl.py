"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_index, bit_reversal_permutation
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _upper_llr_minsum,
    _lower_llr_minsum,
    _prepare_channel_llr,
    f_operation,
)


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first，多项式不含最高位）"""
    reg = np.zeros(crc_length, dtype=int)
    for bit in bits:
        feedback = reg[0] ^ bit
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            for k in range(crc_length):
                if poly[k]:
                    reg[k] ^= feedback
    return reg


_CRC_POLYS = {
    8: np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=int),   # 0x07: x^8+x^2+x+1
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=int),
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    if crc_length not in _CRC_POLYS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_remainder(np.asarray(info_bits, dtype=int), poly, crc_length)
    return np.concatenate([np.asarray(info_bits, dtype=int), remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length not in _CRC_POLYS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    bits = np.asarray(bits, dtype=int)
    poly = _CRC_POLYS[crc_length]
    remainder = _crc_remainder(bits, poly, crc_length)
    return np.all(remainder == 0)


class _Path:
    """单条译码路径（Lazy Copy：数组共享，分裂时复制）"""

    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.active = True

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.active = True
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, paths, l):
        n = self.n
        N = self.N
        for path in paths:
            if not path.active:
                continue
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = _upper_llr_minsum(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        path.L[j, s + 1] = _lower_llr_minsum(
                            path.L[j, s],
                            path.L[j - branch_size, s],
                            int(path.B[j - branch_size, s + 1]),
                        )

    def _update_bits(self, paths, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for path in paths:
            if not path.active:
                continue
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr_val, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr_val >= 0 else 1
        return abs(llr_val) if u_bit != hard else 0.0

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = _prepare_channel_llr(llr_ch)
        N = self.N
        n = self.n

        paths = [_Path(N, n, llr)]
        decoded_bits = {}

        for i in range(N):
            l = bit_reversal_index(i, n)
            self._update_llrs(paths, l)

            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                cur_llr = path.L[l, n]

                if l in self.frozen_indices:
                    pen = self._path_metric_penalty(cur_llr, 0)
                    path.pm += pen
                    path.B[l, n] = 0
                    decoded_bits[l] = 0
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = path.copy()
                        child.pm += self._path_metric_penalty(cur_llr, u_bit)
                        child.B[l, n] = u_bit
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]
            for p in paths:
                p.active = True

            self._update_bits(paths, l)

        # 选择最优路径
        crc_pass_paths = []
        if self.crc_length > 0:
            info_idx = sorted(set(range(N)) - self.frozen_indices)
            for p in paths:
                u_full = p.B[:, n].astype(int)
                info_bits = u_full[info_idx]
                if crc_check(info_bits, self.crc_length):
                    crc_pass_paths.append(p)

        if crc_pass_paths:
            best = min(crc_pass_paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm
