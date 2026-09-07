"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _reorder_llr,
    _sc_decode_core,
    f_operation,
    g_operation,
    sc_decode,
)


def _crc_register_step(reg, bit, poly, crc_length):
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & ((1 << crc_length) - 1)
    if fb:
        reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    for bit in info_bits:
        reg = _crc_register_step(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    for bit in bits:
        reg = _crc_register_step(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

        if list_size == 1 and crc_length == 0:
            self._use_sc = True
        else:
            self._use_sc = False

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        if self._use_sc:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = _reorder_llr(llr_ch)
        N, n = self.N, self.n
        frozen_set = self.frozen_set

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "pm": 0.0,
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        def update_llrs(path, l):
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path["L"][j, s + 1] = f_operation(
                            path["L"][j, s], path["L"][j + branch_size, s]
                        )
                    else:
                        path["L"][j, s + 1] = g_operation(
                            path["L"][j - branch_size, s],
                            path["L"][j, s],
                            path["B"][j - branch_size, s + 1],
                        )

        def update_bits(path, l):
            if l < N // 2:
                return
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path["B"][j - branch_size, s - 1] = (
                            path["B"][j, s] ^ path["B"][j - branch_size, s]
                        )
                        path["B"][j, s - 1] = path["B"][j, s]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                update_llrs(path, l)
                llr0 = path["L"][l, n]

                if l in frozen_set:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._pm_penalty(llr0, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, n] = 0
                    update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._pm_penalty(llr0, u),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][l] = u
                        new_path["B"][l, n] = u
                        update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
