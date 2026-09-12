"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _compute_llr,
    _prepare_channel_llr,
    _s_updater,
    precompute_sc_indices,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _compute_crc_bits(info_bits, crc_length):
    """根据信息比特计算 CRC 校验位（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        reg = (((reg << 1) ^ poly) if (reg & top) else (reg << 1)) & mask
    for _ in range(crc_length):
        reg = (((reg << 1) ^ poly) if (reg & top) else (reg << 1)) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _compute_crc_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = _compute_crc_bits(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（自然序惰性 LLR + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _llr_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = _prepare_channel_llr(llr_ch)
        N = self.N
        n = self.n
        L = self.list_size

        llrs = [np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)]
        s = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
        for path in llrs:
            path[n, :] = llr_ch

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        u_hat_paths = [np.zeros(N, dtype=int) for _ in range(L)]

        for idx in range(N):
            candidates = []
            for path_id in range(L):
                if pm[path_id] == np.inf:
                    continue
                llr_val = _compute_llr(0, idx, llrs[path_id], s[path_id])

                if self.frozen_bits[idx]:
                    new_pm = pm[path_id] + self._llr_penalty(llr_val, 0)
                    candidates.append((new_pm, path_id, 0))
                else:
                    for bit in (0, 1):
                        new_pm = pm[path_id] + self._llr_penalty(llr_val, bit)
                        candidates.append((new_pm, path_id, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[:L]

            new_llrs = [np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)]
            new_s = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
            new_pm = np.full(L, np.inf, dtype=np.float64)
            new_u = [np.zeros(N, dtype=int) for _ in range(L)]

            for new_id, (new_p, old_id, bit) in enumerate(selected):
                new_llrs[new_id] = llrs[old_id].copy()
                new_s[new_id] = s[old_id].copy()
                new_u[new_id] = u_hat_paths[old_id].copy()
                new_pm[new_id] = new_p
                new_u[new_id][idx] = bit
                new_s[new_id][0, idx] = bit

            llrs, s, pm, u_hat_paths = new_llrs, new_s, new_pm, new_u

        paths = [(pm[i], u_hat_paths[i]) for i in range(L) if pm[i] < np.inf]
        if self.crc_length > 0:
            valid = [(p, u) for p, u in paths if crc_check(u, self.crc_length)]
            best = min(valid if valid else paths, key=lambda x: x[0])
        else:
            best = min(paths, key=lambda x: x[0])

        return best[1].copy(), best[0]
