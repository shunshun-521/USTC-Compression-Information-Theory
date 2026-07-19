"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """对 bits + r 个零做模2除法，返回 r 位余数"""
    msg = [int(b) for b in bits] + [0] * crc_length
    for i in range(len(bits)):
        if msg[i]:
            for j in range(crc_length + 1):
                if poly & (1 << (crc_length - j)):
                    msg[i + j] ^= 1
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    return np.array_equal(remainder, bits[-crc_length:])


class Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _update_llrs(self, path, bit_idx):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, bit_idx):
        n, N = self.n, self.N
        if bit_idx < N // 2:
            return
        for s in range(n, n - _active_bit_level(bit_idx, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(bit_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi_nat in range(N):
            bit_idx = _bit_reversed(phi_nat, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, bit_idx)
                llr = path.L[bit_idx, n]

                if self.frozen_bits[phi_nat]:
                    pm = path.pm + self._pm_penalty(llr, 0)
                    child = self._clone_path(path)
                    child.pm = pm
                    child.u_hat[phi_nat] = 0
                    child.B[bit_idx, n] = 0
                    self._update_bits(child, bit_idx)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._pm_penalty(llr, bit)
                        child = self._clone_path(path)
                        child.pm = pm
                        child.u_hat[phi_nat] = bit
                        child.B[bit_idx, n] = bit
                        self._update_bits(child, bit_idx)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = self._select_best_path(paths)
        u_hat = best.u_hat.copy()
        return u_hat, best.pm

    def _clone_path(self, path):
        child = Path(self.N, self.n)
        child.L[:] = path.L
        child.B[:] = path.B
        child.pm = path.pm
        child.u_hat[:] = path.u_hat
        return child

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                payload = p.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, K / N)
    mism = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mism += 1
    print(f"SCL L=1 vs SC mismatch: {mism}/20")
