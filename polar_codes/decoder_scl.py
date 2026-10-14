"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _bit_reversed, _update_llr, _update_bits


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_run(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_run(info_bits, crc_length)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    return _crc_run(np.asarray(bits, dtype=np.int8), crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_add(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llr(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, 0]

                if self.frozen_bits[l]:
                    p0 = path.copy()
                    p0.pm = self._pm_add(p0.pm, llr, 0)
                    p0.B[l, 0] = 0
                    _update_bits(p0.B, l, self.n, self.N)
                    new_paths.append(p0)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = self._pm_add(p.pm, llr, u)
                        p.B[l, 0] = u
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best = paths[0]
        for p in paths:
            u = p.B[:, 0].astype(int)
            if self.crc_length > 0:
                info_idx = np.where(~self.frozen_bits)[0]
                payload = u[info_idx]
                if crc_check(payload, self.crc_length) and (
                    best_crc is None or p.pm < best_crc.pm
                ):
                    best_crc = p
            elif p.pm < best.pm:
                best = p
        if best_crc is not None:
            best = best_crc
        u_hat = best.B[:, 0].astype(int)
        return u_hat, best.pm
