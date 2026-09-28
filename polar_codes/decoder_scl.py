"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from vendor_decoder_utils import active_bit_level, active_llr_level, hard_decision, lower_llr, upper_llr
from vendor_utils import bit_reversed
from decoder_sc import _permute_channel_llr

# ==================== CRC 工具 ====================

def _crc_value(info_bits, crc_length):
    """按比特序列计算 CRC 数值（用于仿真，编解码一致）。"""
    val = 0
    for b in info_bits:
        val = (val << 1) | int(b)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    for _ in range(crc_length):
        if val & (1 << (crc_length - 1)):
            val = ((val << 1) & mask) ^ poly
        else:
            val = (val << 1) & mask
    return val & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_value(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(btm, top, bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _permute_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    new_path = _Path(self.N, self.n)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.pm = path.pm + self._pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = _Path(self.N, self.n)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm = path.pm + self._pm_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
