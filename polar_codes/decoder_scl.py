"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
    precompute_sc_indices,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    register = 0
    for bit in info_bits:
        register ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if register & (1 << (crc_length - 1)):
                register = ((register << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                register = (register << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(register >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed[-crc_length:], bits[-crc_length:])


class Path:
    """单条译码路径"""

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = Path(self.L.shape[0], self.L.shape[1] - 1, None)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, _, _ = precompute_sc_indices(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = self.list_size

        paths = [Path(N, n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                current_llr = path.L[l, n]

                if self.frozen_bits[l]:
                    u = 0
                    new_path = path.copy()
                    new_path.pm = self._path_metric_update(path.pm, current_llr, u)
                    new_path.B[l, n] = u
                    new_path.u_hat[l] = u
                    self._update_bits(new_path, l)
                    new_paths.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm = self._path_metric_update(path.pm, current_llr, u)
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L]

        best_idx = 0
        if self.crc_length > 0:
            crc_pass = []
            for i, p in enumerate(paths):
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_pass.append(i)
            if crc_pass:
                best_idx = min(crc_pass, key=lambda i: paths[i].pm)
            else:
                best_idx = min(range(len(paths)), key=lambda i: paths[i].pm)
        else:
            best_idx = min(range(len(paths)), key=lambda i: paths[i].pm)

        return paths[best_idx].u_hat, paths[best_idx].pm
