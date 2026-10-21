"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _path_penalty(llr, bit):
    """路径度量增量（对数域近似，与 LLR 符号一致时不惩罚）"""
    return 0.0 if (llr >= 0 and bit == 0) or (llr < 0 and bit == 1) else abs(float(llr))


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制决策向量，LLR 子树按需重算）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)

    def _decode_paths(self, llr_ch):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen_bits[index]:
                new_metrics = []
                new_decisions = []
                for path, llr in enumerate(llrs):
                    new_metrics.append(metrics[path] + _path_penalty(float(llr[0]), 0))
                    dec = decisions[path].copy()
                    dec[index] = 0
                    new_decisions.append(dec)
                metrics = new_metrics
                decisions = new_decisions
                return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

            candidates = []
            for path, llr in enumerate(llrs):
                for bit in (0, 1):
                    candidates.append(
                        (metrics[path] + _path_penalty(float(llr[0]), bit), path, bit)
                    )
            candidates.sort(key=lambda x: x[0])
            kept = candidates[: self.list_size]

            new_metrics = []
            new_decisions = []
            betas = []
            parent_map = []
            for metric, path, bit in kept:
                new_metrics.append(metric)
                dec = decisions[path].copy()
                dec[index] = bit
                new_decisions.append(dec)
                betas.append(np.array([bit], dtype=np.int8))
                parent_map.append(path)
            metrics = new_metrics
            decisions = new_decisions
            return betas, parent_map

        def node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)

            half = length // 2
            upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
            beta_upper, map_upper = node(upper, base, half)

            lower_llrs = []
            for p in range(len(map_upper)):
                parent = map_upper[p]
                a = llrs[parent][:half]
                b = llrs[parent][half:]
                lower_llrs.append(g_operation(a, b, beta_upper[p]))

            beta_lower, map_lower = node(lower_llrs, base + half, half)

            beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
            betas = [
                np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
                for p in range(len(beta_lower))
            ]
            parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
            return betas, parent_map

        llr = np.asarray(llr_ch, dtype=np.float64)
        node([llr], 0, self.N)
        return list(zip(metrics, decisions, strict=True))

    def decode(self, llr_ch):
        paths = self._decode_paths(llr_ch)
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for pm, u_hat in paths:
                info_bits = u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.astype(int), pm

        u_hat = paths[0][1]
        return u_hat.astype(int), paths[0][0]
