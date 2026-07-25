"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    prepare_channel_llr,
    bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class _PathState:
  __slots__ = ("pm", "L", "B", "u_hat")

  def __init__(self, N, n, llr):
      self.pm = 0.0
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=np.int8)
      self.L[:, 0] = llr
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _compute_llr(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, n]

    def _update_bits(self, path, l, bit):
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        paths = [_PathState(self.N, self.n, llr)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                llr_bit = self._compute_llr(path, l)

                if l in self.frozen_set:
                    new_path = path
                    new_path.pm += self._metric_penalty(llr_bit, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _PathState(self.N, self.n, llr)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm = path.pm
                        new_path.pm += self._metric_penalty(llr_bit, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, list_size=1):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=list_size).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL L=1 verification passed")
