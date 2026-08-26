"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    register = 0

    for bit in info_bits:
        register ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if register & (1 << (crc_length - 1)):
                register = ((register << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                register = (register << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(register >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class Path:
    """SCL 单条路径（Lazy Copy）。"""

    __slots__ = ("pm", "L", "B", "u_hat", "_cow")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_init
        self.u_hat = np.zeros(N, dtype=int)
        self._cow = False

    def ensure_writable(self):
        if not self._cow:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self._cow = True

    def fork(self):
        child = Path.__new__(Path)
        child.pm = self.pm
        child.L = self.L
        child.B = self.B
        child.u_hat = self.u_hat.copy()
        child._cow = self._cow
        self._cow = False
        return child


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _path_llr(self, path, idx):
        _update_llrs(path.L, path.B, idx, self.n, self.N)
        return path.L[idx, self.n]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _propagate_bit(self, path, idx, bit):
        path.ensure_writable()
        path.u_hat[idx] = bit
        path.B[idx, self.n] = bit
        _update_bits(path.B, idx, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_init = llr_ch[br]

        paths = [Path(self.N, self.n, llr_init)]

        for idx in self.decode_order:
            candidates = []

            for path in paths:
                llr = self._path_llr(path, idx)

                if self.frozen_bits[idx]:
                    child = path.fork()
                    child.pm += self._path_metric_penalty(llr, 0)
                    self._propagate_bit(child, idx, 0)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += self._path_metric_penalty(llr, bit)
                        self._propagate_bit(child, idx, bit)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
