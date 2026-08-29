"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
    _update_llrs,
    _update_bits,
)


CRC_POLYNOMIALS = {
    8: np.array([1, 1, 0, 1, 1, 0, 0, 1, 1], dtype=int),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int),
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = np.zeros(crc_length, dtype=int)
    for bit in np.asarray(bits, dtype=np.int8):
        feedback = int(bit) ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC"""
    return np.all(_crc_remainder(bits, crc_length) == 0)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = self._pm_update(path.pm, llr, 0)
                    child = _Path(self.N, self.n, llr_ch)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = pm
                    child.u_hat = path.u_hat.copy()
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    _update_bits(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = self._pm_update(path.pm, llr, bit)
                        child = _Path(self.N, self.n, llr_ch)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = pm
                        child.u_hat = path.u_hat.copy()
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0 and self.info_indices is not None:
            crc_paths = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            if crc_paths:
                paths = crc_paths

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
