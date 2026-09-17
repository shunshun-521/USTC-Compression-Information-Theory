"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _ms_f


POLY8 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
POLY16 = np.array(
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int
)


def _crc_remainder(bits, poly):
    bits = np.array(bits, dtype=int).copy()
    r = len(poly) - 1
    for i in range(len(bits) - r):
        if bits[i] == 1:
            bits[i : i + r + 1] ^= poly
    return bits[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = POLY8
    elif crc_length == 16:
        poly = POLY16
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    r = crc_length
    padded = np.concatenate([info_bits, np.zeros(r, dtype=int)])
    remainder = _crc_remainder(padded, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class _SCLPath:
    __slots__ = ("pm", "u_hat", "R")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.R = np.zeros((N, n + 1), dtype=np.float64)


class SCLDecoder:
    """SCL 译码器（因子图消息传递 + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, alpha=1.0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.alpha = alpha
        self.large = 1e10
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _compute_llr(self, llr_ch, path, phi):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_ch

        R = path.R.copy()
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large
        for i in range(phi):
            if not self.frozen_bits[i]:
                R[i, 0] = self.large if path.u_hat[i] == 0 else -self.large

        for j in range(self.n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, self.N, 2 * s):
                for k in range(s):
                    idx = i + k
                    idx2 = idx + s
                    L[idx, j - 1] = _ms_f(
                        R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                    )
                    L[idx2, j - 1] = (
                        _ms_f(R[idx, j], L[idx, j], self.alpha) + L[idx2, j]
                    )

        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = self._compute_llr(llr_ch, path, phi)

                if self.frozen_bits[phi]:
                    new_path = _SCLPath(self.N, self.n)
                    new_path.u_hat = path.u_hat.copy()
                    new_path.R = path.R.copy()
                    new_path.pm = path.pm + self._path_metric_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _SCLPath(self.N, self.n)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.R = path.R.copy()
                        new_path.pm = path.pm + self._path_metric_penalty(llr, u_bit)
                        new_path.u_hat[phi] = u_bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.u_hat, best.pm
