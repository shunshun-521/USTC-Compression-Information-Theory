"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

import polar_tree as pt
from decoder_sc import f_operation, g_operation


_CRC_POLY_LOCS = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_remainder(info_bits, crc_length):
    loc = _CRC_POLY_LOCS[crc_length]
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    poly = poly[::-1]

    work = list(int(b) for b in info_bits) + [0] * crc_length
    times = len(work) - crc_length
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= poly[j]
    return np.array(work[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用标准多项式 CRC-8 (0x07) / CRC-16 (0x8005)。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    check = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, check])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = _crc_remainder(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _pm_update(llr_slice, bit_slice):
    """路径度量增量：与 LLR 符号不一致时加 |LLR|。"""
    pm = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


def _f_scalar(La, Lb):
    s1 = 1 if La >= 0 else -1
    s2 = 1 if Lb >= 0 else -1
    return s1 * s2 * min(abs(La), abs(Lb))


def _g_scalar(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.is_info = pt.frozen_to_info_mask(N, self.frozen_bits)
        self.info_indices = np.where(self.is_info)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_value = 0

    def _new_path(self, llr_ch):
        llr_matrix, bit_matrix, _ = pt.init_matrices(self.N)
        llr_matrix[0] = llr_ch
        return {
            "llr": llr_matrix,
            "bit": bit_matrix,
            "pm": 0.0,
            "parent": None,
            "branch_bit": None,
            "split_phi": -1,
        }

    def _run_to_phi(self, path, phi):
        loc = pt.get_up_loc(path["bit"])
        position = [loc[0], loc[1], self.n, self.N]
        path["llr"], path["bit"], _ = pt.sc_tree_step(
            path["llr"],
            path["bit"],
            position,
            self.is_info,
            self.frozen_value,
            _f_scalar,
            _g_scalar,
            stop_pos=phi,
        )

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []

            for pidx, path in enumerate(paths):
                self._run_to_phi(path, phi)

                if not self.is_info[phi]:
                    llr_phi = path["llr"][self.n][phi]
                    bit_val = 0
                    new_pm = path["pm"] + (0.0 if llr_phi >= 0 else abs(llr_phi))
                    new_path = {
                        "llr": path["llr"].copy(),
                        "bit": path["bit"].copy(),
                        "pm": new_pm,
                        "parent": pidx,
                        "branch_bit": bit_val,
                        "split_phi": phi,
                    }
                    new_path["bit"][self.n][phi] = bit_val
                    candidates.append(new_path)
                else:
                    llr_phi = path["llr"][self.n][phi]
                    for bit_val in (0, 1):
                        new_pm = path["pm"] + (
                            0.0 if (llr_phi >= 0) == (bit_val == 0) else abs(llr_phi)
                        )
                        new_path = {
                            "llr": path["llr"].copy(),
                            "bit": path["bit"].copy(),
                            "pm": new_pm,
                            "parent": pidx,
                            "branch_bit": bit_val,
                            "split_phi": phi,
                        }
                        new_path["bit"][self.n][phi] = bit_val
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                u_hat = self._extract_u(path)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, path["pm"]

        best = paths[0]
        return self._extract_u(best), best["pm"]

    def _extract_u(self, path):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            val = path["bit"][self.n][i]
            u_hat[i] = 0 if val == 0 else 1
        return u_hat
