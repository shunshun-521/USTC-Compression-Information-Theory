"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _b_check, _s_updater, _compute_llr


CRC_POLYNOMIALS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int),
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    msg = np.asarray(info_bits, dtype=int).copy()
    r = crc_length
    padded = np.zeros(len(msg) + r, dtype=int)
    padded[: len(msg)] = msg
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i: i + len(poly)] ^= poly
    crc_bits = padded[len(msg): len(msg) + r]
    return np.concatenate([msg, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = CRC_POLYNOMIALS[crc_length]
    data = np.asarray(bits, dtype=int)
    if len(data) < crc_length:
        return False
    msg = data[:-crc_length]
    padded = np.zeros(len(msg) + crc_length, dtype=int)
    padded[: len(msg)] = msg
    for i in range(len(msg)):
        if padded[i] == 1:
            padded[i: i + len(poly)] ^= poly
    expected = padded[len(msg):]
    return np.array_equal(expected, data[-crc_length:])


class _Path:
    __slots__ = ("llrs", "bits", "pm", "u_hat")

    def __init__(self, n, N):
        self.llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
        self.bits = -np.ones((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen = np.asarray(frozen_bits)
        if frozen.dtype != bool:
            frozen = frozen.astype(bool)
        self.frozen = frozen
        self.info_indices = np.where(~self.frozen)[0]
        self._rev = bit_reversal_permutation(N)

    def _init_paths(self, llr_ch):
        path = _Path(self.n, self.N)
        path.llrs[self.n, :] = llr_ch
        return [path]

    def _path_llr(self, path, idx):
        if path.llrs[0, idx] != -np.inf and not (
            self.frozen[idx] and path.llrs[0, idx] == np.inf
        ):
            if not self.frozen[idx]:
                return path.llrs[0, idx]
        path.llrs[0, idx] = _compute_llr(0, idx, path.llrs, path.bits)
        return path.llrs[0, idx]

    def _clone_path(self, path):
        new_path = _Path(self.n, self.N)
        new_path.llrs = path.llrs.copy()
        new_path.bits = path.bits.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._rev]
        paths = self._init_paths(llr_ch)

        for idx in range(self.N):
            candidates = []

            for path in paths:
                llr_val = self._path_llr(path, idx)

                if self.frozen[idx]:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    new_path = self._clone_path(path)
                    new_path.pm += penalty
                    new_path.u_hat[idx] = 0
                    new_path.bits[0, idx] = 0
                    new_path.llrs[0, idx] = np.inf
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        consistent = (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0)
                        if not consistent:
                            new_path.pm += abs(llr_val)
                        new_path.u_hat[idx] = bit
                        new_path.bits[0, idx] = bit
                        new_path.llrs[0, idx] = llr_val
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
