"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr_minsum,
    _upper_llr_exact,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=int)
    n = (len(bits) + 7) // 8
    padded = np.concatenate([bits, np.zeros(n * 8 - len(bits), dtype=int)])
    out = bytearray()
    for i in range(n):
        value = 0
        for j in range(8):
            value = (value << 1) | int(padded[8 * i + j])
        out.append(value)
    return bytes(out)


def _byte_to_bits(value, width=8):
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)


def _crc8_byte(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    if crc_length == 8:
        remainder = _crc8_byte(data)
        crc_bits = _byte_to_bits(remainder, 8)
    else:
        remainder = _crc16_byte(data)
        crc_bits = _byte_to_bits(remainder, 16)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_byte(data) == 0
    return _crc16_byte(data) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


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

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _copy_path(self, src, dst):
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat

    def decode(self, llr_ch):
        """主译码函数。"""
        N = self.N
        n = self.n
        L_size = self.list_size

        llr_internal = np.asarray(llr_ch, dtype=np.float64)

        paths = [_Path(N, n) for _ in range(L_size)]
        paths[0].L[:, 0] = llr_internal
        active = 1

        for phase_idx in range(N):
            l = _bit_reversed(phase_idx, n)

            for pi in range(active):
                path = paths[pi]
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = _upper_llr_exact(
                                path.L[j, s], path.L[j + branch_size, s]
                            )
                        else:
                            path.L[j, s + 1] = _lower_llr_minsum(
                                path.L[j, s],
                                path.L[j - branch_size, s],
                                path.B[j - branch_size, s + 1],
                            )

            current_llrs = [paths[i].L[l, n] for i in range(active)]
            candidates = []

            if l in self.frozen_set:
                for i in range(active):
                    p = paths[i]
                    p.u_hat[l] = 0
                    p.pm += self._pm_penalty(current_llrs[i], 0)
                    p.B[l, n] = 0
                    candidates.append(p)
            else:
                for i in range(active):
                    for u in (0, 1):
                        p = _Path(N, n)
                        self._copy_path(paths[i], p)
                        p.u_hat[l] = u
                        p.pm += self._pm_penalty(current_llrs[i], u)
                        p.B[l, n] = u
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L_size]
            active = len(paths)

            for path in paths:
                if l >= N / 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 2 ** s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                                    path.B[j - branch_size, s]
                                )
                                path.B[j, s - 1] = path.B[j, s]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
