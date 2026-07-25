"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
    _permute_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_internal):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_internal


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _branch_metric(self, llr, bit):
        return 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)

    def _continue_path(self, path, l, bit):
        path.B[l, self.n] = bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr_internal = _permute_llr(llr_ch, N)
        paths = [_PathState(N, n, llr_internal)]

        for l in [bit_reversed(i, n) for i in range(N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = _PathState(N, n, llr_internal)
                    new_path.pm = path.pm + self._branch_metric(llr, 0)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    self._continue_path(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _PathState(N, n, llr_internal)
                        new_path.pm = path.pm + self._branch_metric(llr, bit)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        self._continue_path(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_valid = []
        for path in paths:
            u_hat = path.B[:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append((path.pm, u_hat))
            else:
                crc_valid.append((path.pm, u_hat))

        if self.crc_length > 0:
            passed = [item for item in crc_valid if crc_check(
                item[1][self.info_indices], self.crc_length
            )]
            if passed:
                _, best = min(passed, key=lambda x: x[0])
                return best, min(passed, key=lambda x: x[0])[0]

        best_pm, best = min(crc_valid, key=lambda x: x[0])
        return best, best_pm
