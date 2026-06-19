"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _bit_reversed_index,
    _prepare_channel_llrs,
    _to_frozen_mask,
    _update_bits,
    _update_llrs,
)
from utils import crc_encode as _crc_encode_impl
from utils import crc_check as _crc_check_impl


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    return _crc_encode_impl(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    return _crc_check_impl(bits, crc_length)


def _path_metric_update(pm, llr, bit):
    """路径度量更新：与 LLR 硬判决不一致时增加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=np.int8)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_mask = _to_frozen_mask(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_mask)[0])
        self.info_set = set(np.where(~self.frozen_mask)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llrs(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path.pm, llr, 0)
                    child = self._clone_path(path)
                    child.pm = pm
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = _path_metric_update(path.pm, llr, bit)
                        child = self._clone_path(path)
                        child.pm = pm
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        return self._select_best_path(paths)

    def _clone_path(self, path):
        child = _Path(self.N, self.n, path.L[:, 0])
        child.L[:] = path.L
        child.B[:] = path.B
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        return child

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            info_indices = sorted(self.info_set)
            valid = []
            for path in paths:
                info_bits = path.u_hat[info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
                return best.u_hat, best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat, best.pm


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(1)
    err_l1 = err_sc = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_l1, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc[info_idx], u[info_idx]):
            err_sc += 1
        if not np.array_equal(u_l1[info_idx], u[info_idx]):
            err_l1 += 1
    print(f"L=1 vs SC mismatch: {err_l1}/50 (SC errors {err_sc}/50)")
