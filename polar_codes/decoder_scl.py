"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, f_operation_exact, g_operation_exact,
    bit_reversed_index, _active_llr_level, _active_bit_level,
)

# ==================== CRC 工具 ====================

CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_POLY_BITS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _gf2_crc_remainder(msg_bits, gen_bits):
    """GF(2) 多项式长除法求 CRC 余数"""
    msg = list(msg_bits) + [0] * (len(gen_bits) - 1)
    for i in range(len(msg_bits)):
        if msg[i]:
            for j in range(len(gen_bits)):
                msg[i + j] ^= gen_bits[j]
    return np.array(msg[len(msg_bits):], dtype=int)


def _gf2_crc_verify(bits, gen_bits):
    msg = list(bits)
    for i in range(len(bits) - (len(gen_bits) - 1)):
        if msg[i]:
            for j in range(len(gen_bits)):
                msg[i + j] ^= gen_bits[j]
    return sum(msg[-(len(gen_bits) - 1):]) == 0


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    crc_bits = _gf2_crc_remainder(info_bits, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    gen = CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS
    return _gf2_crc_verify(bits, gen)


def _path_metric_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加惩罚"""
    hard = 0 if llr >= 0 else 1
    penalty = 0.0 if u == hard else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation_exact(
                        path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = _path_metric_update(path.pm, llr, 0)
                    path.B[l, self.n] = 0
                    path.pm = pm
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = self._clone_path(path)
                        child.pm = _path_metric_update(path.pm, llr, u)
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]
            for p in paths:
                p.active = True

        u_hat = paths[0].B[:, self.n].astype(int)
        best_pm = paths[0].pm

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            info_positions = np.where(info_mask)[0]
            valid = []
            for path in paths:
                bits = path.B[:, self.n].astype(int)
                info_bits = bits[info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                valid.sort(key=lambda p: p.pm)
                u_hat = valid[0].B[:, self.n].astype(int)
                best_pm = valid[0].pm

        return u_hat, best_pm

    def _clone_path(self, path):
        """Lazy copy：仅复制必要状态"""
        child = _Path(self.N, self.n, path.L[:, 0])
        child.L[:] = path.L
        child.B[:] = path.B
        child.pm = path.pm
        return child


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)

    # L=1 SCL 应等价于 SC
    mismatches = 0
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC 不一致: {mismatches}/50")
    assert mismatches == 0
