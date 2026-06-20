"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _b_check,
    _compute_llr,
    _update_bits,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial_bits(crc_length):
    if crc_length == 8:
        # x^8 + x^2 + x + 1
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        # x^16 + x^15 + x^2 + 1
        return np.array(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int
        )
    raise ValueError("crc_length 仅支持 8 或 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).flatten()
    poly = _crc_polynomial_bits(crc_length)
    r = crc_length
    msg = np.concatenate([info_bits, np.zeros(r, dtype=int)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i : i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[-r:]])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).flatten()
    poly = _crc_polynomial_bits(crc_length)
    r = crc_length
    if len(bits) < r:
        return False
    data = bits[:-r]
    expected = crc_encode(data, crc_length)[-r:]
    return np.array_equal(bits[-r:], expected)


class _PathState:
  __slots__ = ("llrs", "bits", "pm")

  def __init__(self, n, N):
    self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    self.bits = -np.ones((n + 1, N), dtype=int)
    self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_at(self, path, phi):
        if self.frozen_bits[phi]:
            path.llrs[0, phi] = np.inf
            return path.llrs[0, phi]
        path.llrs[0, phi] = _compute_llr(0, phi, path.llrs, path.bits)
        return path.llrs[0, phi]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, L = self.N, self.list_size

        paths = []
        for _ in range(L):
            p = _PathState(self.n, N)
            p.llrs[self.n, :] = llr_ch.copy()
            p.pm = np.inf
            paths.append(p)
        paths[0].pm = 0.0

        active = 1

        for phi in range(N):
            if self.frozen_bits[phi]:
                for d in range(active):
                    llr_val = self._llr_at(paths[d], phi)
                    paths[d].bits[0, phi] = 0
                    paths[d].pm += self._pm_penalty(llr_val, 0)
            else:
                candidates = []
                for d in range(active):
                    llr_val = self._llr_at(paths[d], phi)
                    for u_bit in (0, 1):
                        pm_new = paths[d].pm + self._pm_penalty(llr_val, u_bit)
                        candidates.append((pm_new, d, u_bit, llr_val))

                candidates.sort(key=lambda x: x[0])
                selected = candidates[:L]

                new_paths = []
                for pm_new, parent_idx, u_bit, llr_val in selected:
                    child = _PathState(self.n, N)
                    child.llrs = paths[parent_idx].llrs.copy()
                    child.bits = paths[parent_idx].bits.copy()
                    child.bits[0, phi] = u_bit
                    child.llrs[0, phi] = llr_val
                    child.pm = pm_new
                    new_paths.append(child)

                paths = new_paths
                active = len(paths)

        u_candidates = [p.bits[0, :].copy() for p in paths]
        pms = np.array([p.pm for p in paths], dtype=np.float64)

        if self.crc_length > 0:
            valid = []
            for idx, u in enumerate(u_candidates):
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(idx)
            if valid:
                best = valid[int(np.argmin(pms[valid]))]
                return u_candidates[best], float(pms[best])

        best = int(np.argmin(pms))
        return u_candidates[best], float(pms[best])
