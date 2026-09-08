"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SC
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in info_bits:
        feedback = (bit + reg[0]) % 2
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg = (reg + poly[1:]) % 2
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        feedback = (bit + reg[0]) % 2
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg = (reg + poly[1:]) % 2
    return np.all(reg == 0)


class SCLDecoder:
    """SCL 译码器（Permuted SC + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_state(self, L, B, pm, u_hat):
        return L.copy(), B.copy(), pm, u_hat.copy()

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = []
        L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append((L0, B0, 0.0, np.zeros(self.N, dtype=int)))

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []

            for L, B, pm, u_hat in paths:
                self._update_llrs(L, B, l)
                llr = L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    B[l, self.n] = 0
                    u_hat[l] = 0
                    self._update_bits(B, l)
                    new_paths.append((L, B, pm + pen, u_hat))
                else:
                    for u in (0, 1):
                        Lc, Bc, pmc, uhc = self._copy_state(L, B, pm, u_hat)
                        pmc += self._pm_penalty(llr, u)
                        Bc[l, self.n] = u
                        uhc[l] = u
                        self._update_bits(Bc, l)
                        new_paths.append((Lc, Bc, pmc, uhc))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_ok = []
            for _, _, pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append((pm, u_hat))
            if crc_ok:
                best = min(crc_ok, key=lambda x: x[0])[1]
            else:
                best = paths[0][3]
        else:
            best = paths[0][3]

        return best.copy(), paths[0][2]
