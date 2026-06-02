"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for bit in info_bits:
            reg ^= (bit << 7)
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= (bit << 15)
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（递归列表译码，与 SC 树结构一致）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_ind < 0.5)[0]

    @staticmethod
    def _pm_add(pm, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return pm + (0.0 if u_bit == u_from_llr else abs(llr))

    def _list_decode(self, llr, frozen, pm_base=0.0):
        """返回 [(pm, u_hat, u_up)] 列表，u_hat/u_up 为当前子树长度"""
        n = len(llr)
        if n == 1:
            results = []
            if frozen[0] >= 0.5:
                u = np.array([0.0])
                results.append((self._pm_add(pm_base, llr[0], 0), u, u.copy()))
            else:
                for u_bit in (0.0, 1.0):
                    u = np.array([u_bit])
                    results.append((self._pm_add(pm_base, llr[0], int(u_bit)), u, u.copy()))
            return results

        half = n // 2
        llr_left = llr[:half]
        llr_right = llr[half:]
        frozen_left = frozen[:half]
        frozen_right = frozen[half:]

        llr_upper = f_operation(llr_left, llr_right)
        left_results = self._list_decode(llr_upper, frozen_left, pm_base)

        all_results = []
        for pm_l, u_l, u_l_up in left_results:
            llr_lower = g_operation(llr_left, llr_right, u_l_up)
            right_results = self._list_decode(llr_lower, frozen_right, pm_l)
            for pm_r, u_r, u_r_up in right_results:
                u_hat = np.concatenate([u_l, u_r])
                u_l_up_xor = (u_l_up.astype(int) ^ u_r_up.astype(int)).astype(np.float64)
                u_up = np.concatenate([u_l_up_xor, u_r_up])
                all_results.append((pm_r, u_hat, u_up))

        all_results.sort(key=lambda x: x[0])
        return all_results[: self.list_size]

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._list_decode(llr_ch, self.frozen_ind)

        if self.crc_length > 0:
            valid = []
            for pm, u_hat, _ in paths:
                info_bits = u_hat[self.info_indices].astype(int)
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                best = min(valid, key=lambda x: x[0])
            else:
                best = min(paths, key=lambda x: x[0])
                best = (best[0], best[1])
        else:
            pm, u_hat, _ = min(paths, key=lambda x: x[0])
            best = (pm, u_hat)

        return best[1].astype(int), best[0]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 5.0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    err_sc = err_scl = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
            err_sc += 1
        if not np.array_equal(
            SCLDecoder(N, frozen_bits, list_size=1).decode(llr)[0][info_idx], u[info_idx]
        ):
            err_scl += 1
    print(f"L=1 vs SC: sc_err={err_sc}, scl_err={err_scl}")
