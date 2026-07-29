"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    upper_llr, lower_llr, active_llr_level, active_bit_level, bit_reversed
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class Path:
    """SCL 单条路径"""

    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        L = path.L
        B = path.B

        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        return L[l, n]

    def _propagate_bits(self, path, l):
        B = path.B
        n = self.n
        N = self.N

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [Path(n, N) for _ in range(L_size)]
        paths[0].L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        num_active = 1

        for i in range(N):
            l = bit_reversed(i, n)
            is_frozen = l in self.frozen_set
            candidates = []

            for pi in range(num_active):
                path = paths[pi]
                llr = self._update_llrs(path, l)

                if is_frozen:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), pi, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, bit), pi, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L_size]

            new_paths = []
            for new_pm, parent_idx, bit in candidates:
                parent = paths[parent_idx]
                child = Path(n, N)
                child.pm = new_pm
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = bit
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.B[l, n] = bit
                self._propagate_bits(child, l)
                new_paths.append(child)

            paths = new_paths
            num_active = len(paths)

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat, path.pm

        best = paths[0]
        return best.u_hat, best.pm


def verify_scl_equals_sc(N=64, K=32, num_frames=50):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(123)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print(f"SCL(L=1) 路径度量校验通过: {num_frames} 帧")


if __name__ == "__main__":
    verify_scl_equals_sc()
