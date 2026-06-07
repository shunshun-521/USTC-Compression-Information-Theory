"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _b_check,
    _compute_llr,
    _s_updater,
    f_operation,
    g_operation,
)

# ==================== CRC 工具 ====================

_CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
_CRC16_POLY_BITS = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int
)


def _crc_remainder(bits, poly):
    bits = np.asarray(bits, dtype=int)
    poly = np.asarray(poly, dtype=int)
    work = np.zeros(len(bits) + len(poly) - 1, dtype=int)
    work[: len(bits)] = bits
    for idx in range(len(work) - len(poly) + 1):
        if work[idx] == 1:
            work[idx : idx + len(poly)] ^= poly
    return work[-(len(poly) - 1) :]


def _crc_verify(bits, poly):
    work = np.asarray(bits, dtype=int).copy()
    poly = np.asarray(poly, dtype=int)
    for idx in range(len(work) - len(poly) + 1):
        if work[idx] == 1:
            work[idx : idx + len(poly)] ^= poly
    return np.all(work[-(len(poly) - 1) :] == 0)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    remainder = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    return _crc_verify(bits, poly)


# ==================== SCL 译码器 ====================


class _PathState:
  __slots__ = ("llrs", "partial_sums", "pm", "u_hat")

  def __init__(self, n, N):
      self.llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
      self.partial_sums = -np.ones((n + 1, N), dtype=np.int32)
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（惰性 LLR 复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_bits = ~self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.info_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone_path(self, path):
        new_path = _PathState(self.n, self.N)
        new_path.llrs[self.n, :] = path.llrs[self.n, :]
        new_path.partial_sums = path.partial_sums.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _get_llr(self, path, phi):
        if self.frozen_bits[phi]:
            return np.inf
        return _compute_llr(0, phi, path.llrs, path.partial_sums)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        path = _PathState(self.n, self.N)
        path.llrs[self.n, :] = llr_ch
        paths = [path]

        for phi in range(self.N):
            new_paths = []
            for p in paths:
                llr = self._get_llr(p, phi)
                if self.frozen_bits[phi]:
                    penalty = self._path_metric_penalty(llr, 0)
                    p.pm += penalty
                    p.u_hat[phi] = 0
                    p.partial_sums[0, phi] = 0
                    p.llrs[0, phi] = np.inf
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        cp = self._clone_path(p)
                        cp.pm += self._path_metric_penalty(llr, bit)
                        cp.u_hat[phi] = bit
                        cp.partial_sums[0, phi] = bit
                        cp.llrs[0, phi] = llr
                        new_paths.append(cp)

            new_paths.sort(key=lambda item: item.pm)
            paths = new_paths[: self.list_size]

        best_idx = 0
        if self.crc_length > 0:
            passed = []
            for idx, p in enumerate(paths):
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    passed.append(idx)
            if passed:
                best_idx = min(passed, key=lambda i: paths[i].pm)

        best = paths[best_idx]
        return best.u_hat.copy(), best.pm
