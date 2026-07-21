"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _metric_penalty(llr, bit):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if bit == preferred else abs(llr)


class SCLDecoder:
    """SCL 译码器（预分配数组，减少 Python 开销）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _decode_paths(self, pms, u_hats, u_ups, llr, frozen_seg, offset, n_paths):
        n = len(llr)
        if n == 1:
            idx = offset
            out_pm = []
            out_u = []
            out_up = []
            for p in range(n_paths):
                llr0 = llr[0]
                if self.frozen_bits[idx]:
                    out_pm.append(pms[p] + _metric_penalty(llr0, 0))
                    u = u_hats[p].copy()
                    u[idx] = 0
                    out_u.append(u)
                    out_up.append(np.array([0], dtype=np.int8))
                else:
                    for bit in (0, 1):
                        out_pm.append(pms[p] + _metric_penalty(llr0, bit))
                        u = u_hats[p].copy()
                        u[idx] = bit
                        out_u.append(u)
                        out_up.append(np.array([bit], dtype=np.int8))
            order = np.argsort(out_pm)
            keep = order[: self.list_size]
            return (
                [out_pm[i] for i in keep],
                [out_u[i] for i in keep],
                [out_up[i] for i in keep],
            )

        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        llr_left = f_operation(llr1, llr2)

        up_pms, up_u, up_up = self._decode_paths(
            pms, u_hats, u_ups, llr_left, frozen_seg[:half], offset, n_paths
        )

        all_pm, all_u, all_up = [], [], []
        for i in range(len(up_pms)):
            llr_right = g_operation(llr1, llr2, up_up[i])
            lo_pms, lo_u, lo_up = self._decode_paths(
                [up_pms[i]], [up_u[i]], [up_up[i]], llr_right, frozen_seg[half:], offset + half, 1
            )
            for j in range(len(lo_pms)):
                all_pm.append(lo_pms[j])
                all_u.append(lo_u[j])
                merged_up = np.concatenate([np.bitwise_xor(up_up[i], lo_up[j]), lo_up[j]])
                all_up.append(merged_up)

        order = np.argsort(all_pm)
        keep = order[: self.list_size]
        return (
            [all_pm[i] for i in keep],
            [all_u[i] for i in keep],
            [all_up[i] for i in keep],
        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u0 = np.zeros(self.N, dtype=np.int8)
        pms, u_hats, u_ups = self._decode_paths(
            [0.0], [u0], [np.array([], dtype=np.int8)], llr_ch, self.frozen_bits, 0, 1
        )

        if self.crc_length > 0:
            best_idx = 0
            best_pm = float('inf')
            found = False
            for i, u in enumerate(u_hats):
                if crc_check(u[self.info_indices], self.crc_length):
                    if pms[i] < best_pm:
                        best_pm = pms[i]
                        best_idx = i
                        found = True
            if not found:
                best_idx = 0
        else:
            best_idx = 0

        return u_hats[best_idx].astype(int), float(pms[best_idx])
