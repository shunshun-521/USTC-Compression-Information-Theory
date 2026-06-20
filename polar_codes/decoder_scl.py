"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_compute(bits, poly, crc_length, append_zeros=False):
    """CRC 模2除法。"""
    data = np.concatenate([bits, np.zeros(crc_length, dtype=int)]) if append_zeros else bits
    reg = 0
    for bit in data:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_compute(info_bits, poly, crc_length, append_zeros=True)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_compute(bits, poly, crc_length, append_zeros=False) == 0


class _Path:
    """单条译码路径。"""

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)

    def copy(self):
        n = self.L.shape[1] - 1
        child = _Path(self.L.shape[0], n)
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child.L = self.L.copy()
        child.B = self.B.copy()
        return child


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def _init_llrs(self, path, llr_ch):
        for j in range(self.N):
            path.L[j, 0] = llr_ch[_bit_reversed(j, self.n)]

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        paths = [_Path(N, n)]
        self._init_llrs(paths[0], llr_ch)

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_path = path.copy()
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, n] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        llr_bit = 0 if llr_val >= 0 else 1
                        if bit != llr_bit:
                            new_path.pm += abs(llr_val)
                        new_path.u_hat[l] = bit
                        new_path.B[l, n] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

            for path in paths:
                self._update_bits(path, l)

        if self.crc_length > 0:
            valid = [p for p in paths
                     if crc_check(p.u_hat[self.info_positions], self.crc_length)]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.copy(), chosen.pm
