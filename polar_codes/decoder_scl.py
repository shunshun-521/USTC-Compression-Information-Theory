"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _get_llr, INF


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        return np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int
        )
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    reg = np.zeros(crc_length, dtype=int)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _PathState:
    __slots__ = ("llrs", "bits", "pm", "u_hat")

    def __init__(self, n, N):
        self.llrs = np.full((n + 1, N), -INF, dtype=np.float64)
        self.bits = np.full((n + 1, N), -1, dtype=np.int64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone(self, path):
        new_path = _PathState(self.n, self.N)
        new_path.llrs = path.llrs.copy()
        new_path.bits = path.bits.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.n, self.N)]
        paths[0].llrs[self.n, :] = llr_ch

        for phi in range(self.N):
            candidates = []
            for path in paths:
                if self.frozen_bits[phi]:
                    llr = _get_llr(0, phi, path.llrs, path.bits)
                    new_path = self._clone(path)
                    penalty = abs(llr) if llr < 0 else 0.0
                    new_path.pm += penalty
                    new_path.u_hat[phi] = 0
                    new_path.bits[0, phi] = 0
                    new_path.llrs[0, phi] = INF
                    candidates.append(new_path)
                else:
                    llr = _get_llr(0, phi, path.llrs, path.bits)
                    for bit in (0, 1):
                        new_path = self._clone(path)
                        hard = 0 if llr >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr)
                        new_path.pm += penalty
                        new_path.u_hat[phi] = bit
                        new_path.bits[0, phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
