"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _Li, _f_node_minsum


def _crc_remainder(bits, poly):
    """计算 CRC 余数（标准多项式长除法，内部自动补零）"""
    bits = np.asarray(bits, dtype=int)
    m = len(poly)
    data = np.concatenate([bits, np.zeros(m - 1, dtype=int)])
    n = len(bits)
    for i in range(n):
        if data[i]:
            data[i:i + m] ^= poly
    return data[-(m - 1):]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    elif crc_length == 16:
        poly = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            dtype=int,
        )
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    else:
        poly = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            dtype=int,
        )
    rem = _crc_remainder(bits, poly)
    return np.all(rem == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _if_info(self):
        """polarLib 风格：1=信息位，0=冻结位"""
        return (self.frozen_bits == 0).astype(np.int32)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size
        if_info = self._if_info()

        llrs = [np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)]
        for dd in range(L):
            llrs[dd][n, :] = llr_ch

        s = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
        PM = np.full(L, np.inf, dtype=np.float64)
        PM[0] = 0.0
        PM_DM = np.zeros(2 * L, dtype=np.float64)

        for ii in range(N):
            DM = np.zeros(L, dtype=np.float64)

            if if_info[ii] == 0:
                for dd in range(L):
                    llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                    s[dd][0, ii] = 0
                    PM[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
            else:
                for dd in range(L):
                    llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                    s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                    DM[dd] = np.abs(llrs[dd][0, ii])

                if L > 1:
                    PM_DM[:L] = PM
                    PM_DM[L:] = PM + DM
                    idx_sort = np.argsort(PM_DM)
                    idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                    idx_min_up = idx_sort[L:][idx_sort[L:] < L]
                    for bb in range(len(idx_min_low)):
                        up, low = idx_min_up[bb], idx_min_low[bb]
                        llrs[up] = np.copy(llrs[low])
                        s[up] = np.copy(s[low])
                        s[up][0, ii] = 1 - s[low][0, ii]
                        PM[up] = PM_DM[low + L]

        u_candidates = np.array([s[dd][0, :] for dd in range(L)], dtype=int)

        if self.crc_length > 0 and len(self.info_indices) > self.crc_length:
            k_crc = len(self.info_indices) - self.crc_length
            info_idx = self.info_indices
            passed = []
            for dd in range(L):
                info_bits = u_candidates[dd, info_idx]
                payload = info_bits[:k_crc]
                crc_part = info_bits[k_crc:]
                expected = crc_encode(payload, self.crc_length)
                if np.array_equal(crc_part, expected[-self.crc_length:]):
                    passed.append(dd)
            if passed:
                best = passed[np.argmin(PM[passed])]
                return u_candidates[best], PM[best]

        best = int(np.argmin(PM))
        return u_candidates[best], PM[best]
