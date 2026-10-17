"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode


# ==================== CRC 工具 ====================

_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLYS[crc_length]
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


def _scl_decode_tree(llr_ch, frozen_ind, list_size):
    """递归 SCL（与 SC 树结构一致）。"""

    def decode_segment(llr_seg, frozen_seg):
        n = len(llr_seg)
        if n > 1:
            half = n // 2
            llr1, llr2 = llr_seg[:half], llr_seg[half:]
            f1, f2 = frozen_seg[:half], frozen_seg[half:]

            x_llr1 = f_operation(llr1, llr2)
            paths1 = decode_segment(x_llr1, f1)

            all_paths = []
            for pm1, u1, up1 in paths1:
                x_llr2 = g_operation(llr1, llr2, up1)
                paths2 = decode_segment(x_llr2, f2)
                for pm2, u2, up2 in paths2:
                    pm = pm1 + pm2
                    u_hat = np.concatenate([u1, u2])
                    up_xor = (up1.astype(np.int8) ^ up2.astype(np.int8)).astype(np.float64)
                    u_up = np.concatenate([up_xor, up2])
                    all_paths.append((pm, u_hat, u_up))

            all_paths.sort(key=lambda x: x[0])
            return all_paths[:list_size]

        # 叶节点
        llr = float(llr_seg[0])
        if frozen_seg[0] >= 0.5:
            pm = _pm_update(0.0, llr, 0)
            u = np.array([0.0])
            return [(pm, u, u)]

        p0 = _pm_update(0.0, llr, 0)
        p1 = _pm_update(0.0, llr, 1)
        paths = [
            (p0, np.array([0.0]), np.array([0.0])),
            (p1, np.array([1.0]), np.array([1.0])),
        ]
        paths.sort(key=lambda x: x[0])
        return paths[:list_size]

    return decode_segment(llr_ch, frozen_ind)


class SCLDecoder:
    """SCL 译码器（列表大小 L，可选 CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        frozen_ind = self.frozen_bits.astype(np.float64)
        paths = _scl_decode_tree(llr_ch, frozen_ind, self.list_size)
        best_pm, best_u, _ = paths[0]
        best_u = best_u.astype(int)

        if self.crc_length > 0:
            valid = []
            for pm, u, _ in paths:
                bits = u.astype(int)[self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append((pm, u.astype(int)))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])

        return best_u, best_pm


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    """L=1 时应与 SC 一致。"""
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
