"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _li, _b_check, INF


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    elif crc_length == 16:
        poly = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int
        )
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg_len = len(poly) - 1
    reg = np.zeros(reg_len, dtype=int)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]

    crc_bits = reg
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded[-crc_length:], bits[-crc_length:])


def _info_mask_from_frozen(frozen_bits):
    """frozen_bits: 1=冻结 -> info_mask: 1=信息"""
    fb = np.asarray(frozen_bits, dtype=int)
    return (1 - fb).astype(int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_mask = _info_mask_from_frozen(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = []
        bits_list = []
        for _ in range(L):
            llrs = np.full((n + 1, N), -INF, dtype=np.float64)
            bits = np.full((n + 1, N), -1, dtype=int)
            llrs[n, :] = llr_ch
            llrs_list.append(llrs)
            bits_list.append(bits)

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for i in range(N):
            dm = np.zeros(L, dtype=np.float64)

            for path in range(L):
                llrs = llrs_list[path]
                bits = bits_list[path]

                if self.frozen_bits[i] == 1:
                    _li(0, i, llrs, bits, n)
                    bits[0, i] = 0
                    llrs[0, i] = INF
                    pm[path] += max(0.0, -llrs[0, i]) if llrs[0, i] < INF else 0.0
                else:
                    llr_val = _li(0, i, llrs, bits, n)
                    decision = 1 if llr_val < 0 else 0
                    bits[0, i] = decision
                    dm[path] = abs(llr_val)

            if self.frozen_bits[i] == 0 and L > 1:
                pm_dm = np.concatenate([pm, pm + dm])
                idx_sort = np.argsort(pm_dm)
                idx_low = idx_sort[:L]
                idx_high = idx_sort[L:]

                new_llrs = []
                new_bits = []
                new_pm = np.zeros(L, dtype=np.float64)

                for new_p, old_idx in enumerate(idx_low):
                    if old_idx < L:
                        new_llrs.append(llrs_list[old_idx])
                        new_bits.append(bits_list[old_idx])
                        new_pm[new_p] = pm[old_idx]
                    else:
                        src = old_idx - L
                        llrs_copy = llrs_list[src].copy()
                        bits_copy = bits_list[src].copy()
                        bits_copy[0, i] = 1 - bits_list[src][0, i]
                        new_llrs.append(llrs_copy)
                        new_bits.append(bits_copy)
                        new_pm[new_p] = pm[src] + dm[src]

                llrs_list = new_llrs
                bits_list = new_bits
                pm = new_pm

        info_indices = np.where(self.info_mask == 1)[0]

        if self.crc_length > 0:
            valid_paths = []
            for path in range(L):
                u = bits_list[path][0, :]
                info_bits = u[info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid_paths.append(path)
            if valid_paths:
                best = valid_paths[np.argmin(pm[valid_paths])]
            else:
                best = int(np.argmin(pm))
        else:
            best = int(np.argmin(pm))

        u_hat = bits_list[best][0, :].astype(int)
        return u_hat, pm[best]
