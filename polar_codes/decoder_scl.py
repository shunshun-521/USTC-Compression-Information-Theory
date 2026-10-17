"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _upper_llr,
    f_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan, dtype=np.float64),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi_nat in range(N):
            l = _bit_reversed_index(phi_nat, n)
            new_paths = []

            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                L_work = L.copy()
                B_work = B.copy()

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L_work[j, s + 1] = _upper_llr(L_work[j, s], L_work[j + branch_size, s])
                        else:
                            top_bit = int(B_work[j - branch_size, s + 1])
                            L_work[j, s + 1] = _lower_llr(
                                L_work[j, s], L_work[j - branch_size, s], top_bit
                            )

                llr_bit = L_work[l, n]
                if l in self.frozen_set:
                    u = 0
                    pm_new = self._path_metric_update(pm, llr_bit, u)
                    B_work[l, n] = u
                    if l >= N / 2:
                        for s in range(n, n - _active_bit_level(l, n), -1):
                            block_size = 2 ** s
                            branch_size = block_size // 2
                            for j in range(l, -1, -block_size):
                                if j % block_size >= branch_size:
                                    B_work[j - branch_size, s - 1] = int(B_work[j, s]) ^ int(
                                        B_work[j - branch_size, s]
                                    )
                                    B_work[j, s - 1] = B_work[j, s]
                    new_paths.append({"L": L_work, "B": B_work, "pm": pm_new})
                else:
                    for u in (0, 1):
                        Lc = L_work.copy()
                        Bc = B_work.copy()
                        pm_new = self._path_metric_update(pm, llr_bit, u)
                        Bc[l, n] = u
                        if l >= N / 2:
                            for s in range(n, n - _active_bit_level(l, n), -1):
                                block_size = 2 ** s
                                branch_size = block_size // 2
                                for j in range(l, -1, -block_size):
                                    if j % block_size >= branch_size:
                                        Bc[j - branch_size, s - 1] = int(Bc[j, s]) ^ int(
                                            Bc[j - branch_size, s]
                                        )
                                        Bc[j, s - 1] = Bc[j, s]
                        new_paths.append({"L": Lc, "B": Bc, "pm": pm_new})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        u_candidates = [p["B"][:, n].astype(int) for p in paths]
        pms = [p["pm"] for p in paths]

        if self.crc_length > 0:
            valid = []
            for u_hat, pm in zip(u_candidates, pms):
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        best = int(np.argmin(pms))
        return u_candidates[best], pms[best]
