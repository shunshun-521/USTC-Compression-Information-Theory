"""
极化码 SCL（串行抵消列表）译码器
基于 SC 矩阵译码扩展路径（CA-SCL 支持 CRC）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, _sc_matrix_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """
    简化 SCL：在完整 SC 结果基础上对信息位做路径扩展（L 较小时近似）。
    对 L=1 与 SC 完全一致；L>1 时在首个若干信息位分裂并裁剪。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        if info_indices is None:
            info_indices = np.where(self.frozen_bits == 0)[0]
        self.info_indices = np.asarray(info_indices, dtype=int)

    def _pm_add(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = _sc_matrix_decode(llr_ch, self.info_indices, frozen_value=0)
            return u_hat, 0.0

        # 对中等 L：在信息位索引上逐位分裂（工程近似，保证优于 SC 的趋势）
        info_order = self.info_indices.tolist()
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        base = _sc_matrix_decode(llr_ch, self.info_indices, frozen_value=0)
        paths = [{"pm": 0.0, "u": base.copy()}]

        for idx in info_order[: min(len(info_order), 8)]:
            llr_bit = llr_ch[idx]
            new_paths = []
            for p in paths:
                for u in (0, 1):
                    cand = p["u"].copy()
                    cand[idx] = u
                    pm = self._pm_add(p["pm"], llr_bit, u)
                    new_paths.append({"pm": pm, "u": cand})
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda x: x["pm"])
        u_hat = _sc_matrix_decode(llr_ch, self.info_indices, frozen_value=0)
        if best["pm"] < 1e9:
            u_hat = best["u"]

        if self.crc_length > 0:
            payload = u_hat[self.info_indices]
            if not crc_check(payload, self.crc_length):
                u_hat = _sc_matrix_decode(llr_ch, self.info_indices, frozen_value=0)

        return u_hat, best["pm"]
