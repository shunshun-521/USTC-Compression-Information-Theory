"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _active_bit_level, _active_llr_level, _update_bits, f_operation, g_operation
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    crc = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        if crc & msb:
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


def _path_llr(path, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, path.L.shape[0], block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                path.L[j, s + 1] = g_operation(
                    path.L[j - branch_size, s],
                    path.L[j, s],
                    path.B[j - branch_size, s + 1],
                )
    return path.L[l, n]


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N, llr_ch=None):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N, llr_ch.copy())]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                llr = _path_llr(path, l, self.n)
                if self.frozen_bits[l]:
                    candidates.append((_pm_update(path.pm, llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((_pm_update(path.pm, llr, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, pidx, u_bit in candidates:
                parent = paths[pidx]
                child = _Path(self.n, self.N)
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.u_hat[:] = parent.u_hat
                child.pm = new_pm
                child.u_hat[l] = u_bit
                child.B[l, self.n] = u_bit
                _update_bits(child.B, l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
