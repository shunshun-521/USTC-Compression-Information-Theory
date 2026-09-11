"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class Path:
    """SCL 单条路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [int(self.br[i]) for i in range(N)]

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llrs(self, path, phi):
        for s in range(self.n - active_llr_level(phi, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phi, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, phi, bit):
        path.B[phi, self.n] = bit
        if phi >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(phi, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(phi, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [Path(self.N, self.n, llr_ch.copy())]

        for phi in self.decode_order:
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, phi)
                cur_llr = path.L[phi, self.n]

                if self.frozen_bits[phi]:
                    pm = path.pm + self._llr_penalty(cur_llr, 0)
                    child = Path(self.N, self.n, path.L[:, 0].copy())
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = pm
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[phi] = 0
                    self._update_bits(child, phi, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._llr_penalty(cur_llr, bit)
                        child = Path(self.N, self.n, path.L[:, 0].copy())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = pm
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[phi] = bit
                        self._update_bits(child, phi, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(8.0, K / N)
    scl_err = sc_err = 0
    for seed in range(50):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc[info_idx], u[info_idx]):
            sc_err += 1
        if not np.array_equal(u_scl[info_idx], u[info_idx]):
            scl_err += 1
    print(f"L=1 vs SC: sc_err={sc_err}, scl_err={scl_err} /50")
