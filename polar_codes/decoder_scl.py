"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    compute_sc_llr_at_phase,
    g_operation,
    propagate_bit_sc,
    sc_decode,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    register = 0

    for bit in info_bits:
        register <<= 1
        register |= int(bit)
        if register & (1 << crc_length):
            register ^= poly

    crc_bits = np.zeros(crc_length, dtype=np.int8)
    for i in range(crc_length - 1, -1, -1):
        crc_bits[crc_length - 1 - i] = (register >> i) & 1

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def _clone_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phase in range(self.N):
            l = self.decode_order[phase]
            candidates = []

            for path in paths:
                llr = compute_sc_llr_at_phase(path["L"], path["B"], l, self.n)

                if l in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path["pm"] = self._path_metric_update(path["pm"], llr, 0)
                    new_path["u_hat"][l] = 0
                    propagate_bit_sc(new_path["B"], l, 0, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path["pm"] = self._path_metric_update(path["pm"], llr, u_bit)
                        new_path["u_hat"][l] = u_bit
                        propagate_bit_sc(new_path["B"], l, u_bit, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:self.list_size]

        crc_pass = []
        for path in paths:
            info_bits = path["u_hat"][self.info_indices]
            if self.crc_length == 0 or crc_check(info_bits, self.crc_length):
                crc_pass.append(path)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
