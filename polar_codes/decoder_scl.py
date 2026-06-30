"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
    prepare_channel_llr,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[crc_length:])


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u == hard:
        return pm
    return pm + abs(llr)


class _Path:
    """单条 SCL 路径，Lazy Copy 通过 copy-on-write 实现。"""

    __slots__ = ("L", "B", "pm", "u_hat", "shared")

    def __init__(self, N, n, llr_ch, shared=None):
        if shared is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.float64)
            self.L[:, 0] = llr_ch
            self.shared = None
        else:
            self.L = shared.L
            self.B = shared.B
            self.shared = shared
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def fork(self):
        child = _Path(0, 0, None, shared=self)
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        return child

    def ensure_owned(self):
        if self.shared is not None:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.shared = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                path.ensure_owned()
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    u = 0
                    path.pm = _path_metric_update(path.pm, llr, u)
                    path.u_hat[l] = u
                    path.B[l, n] = u
                    _update_bits(path.B, l, n)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = path.fork()
                        child.ensure_owned()
                        child.pm = _path_metric_update(path.pm, llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        _update_bits(child.B, l, n)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
