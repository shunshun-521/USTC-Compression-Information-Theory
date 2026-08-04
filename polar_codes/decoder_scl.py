"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed, active_llr_level, active_bit_level,
    f_operation, g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8+x^2+x+1), CRC-16: 0x8005
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
        reg ^= (bit << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class Path:
    """SCL 译码单条路径"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_layers(self, l):
        return list(range(self.n - active_llr_level(l, self.n), self.n))

    def _path_bit_layers(self, l):
        if l < self.N // 2:
            return []
        return list(range(self.n, self.n - active_bit_level(l, self.n), -1))

    def _update_llrs(self, path, l):
        for s in self._path_llr_layers(l):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        for s in self._path_bit_layers(l):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u_bit == preferred else abs(llr)

    def decode(self, llr_ch):
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for phi_nat in range(self.N):
            l = bit_reversed(phi_nat, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_dec = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    new_pm = path.pm + self._pm_penalty(llr_dec, 0)
                    candidates.append((new_pm, path, u_bit))
                else:
                    for u_bit in (0, 1):
                        new_pm = path.pm + self._pm_penalty(llr_dec, u_bit)
                        candidates.append((new_pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[:self.list_size]

            new_paths = []
            for pm, parent, u_bit in selected:
                child = Path(self.N, self.n)
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.u_hat[:] = parent.u_hat
                child.pm = pm
                child.B[l, self.n] = u_bit
                child.u_hat[l] = u_bit
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        # 选择最优路径
        best_path = None
        best_pm = float("inf")

        if self.crc_length > 0:
            crc_valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
            pool = crc_valid if crc_valid else paths
        else:
            pool = paths

        for path in pool:
            if path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        return best_path.u_hat.copy(), best_pm


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=bool)
    frozen_bits[np.setdiff1d(np.arange(N), info_idx)] = True
    sigma = eb_n0_to_sigma(10.0, K / N)

    # L=1 应等价于 SC
    errors = 0
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x) + np.random.normal(0, sigma, N), sigma)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_scl, u_sc):
            errors += 1
    print(f"SCL L=1 vs SC: {errors} mismatches in 50 frames")
