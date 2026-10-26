"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int) % 2
    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int) % 2
    if bits.size < crc_length:
        return False
    payload = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], crc_encode(payload, crc_length)[-crc_length:])


def _path_penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（路径度量 + 列表裁剪；L=1 时等价于 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_positions = np.flatnonzero(~self.frozen_bits)

    def _scl_decode_all(self, channel_llr):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=int)]

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen_bits[index]:
                for path, llr in enumerate(llrs):
                    metrics[path] += _path_penalty(float(llr[0]), 0)
                    decisions[path][index] = 0
                return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

            candidates = [
                (metrics[path] + _path_penalty(float(llr[0]), bit), path, bit)
                for path, llr in enumerate(llrs)
                for bit in (0, 1)
            ]
            candidates.sort(key=lambda c: c[0])
            kept = candidates[: self.list_size]
            new_metrics, new_decisions, betas, parent_map = [], [], [], []
            for metric, path, bit in kept:
                new_metrics.append(metric)
                d = decisions[path].copy()
                d[index] = bit
                new_decisions.append(d)
                betas.append(np.array([bit], dtype=int))
                parent_map.append(path)
            metrics[:] = new_metrics
            decisions[:] = new_decisions
            return betas, parent_map

        def node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)
            half = length // 2
            upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
            beta_u, map_u = node(upper, base, half)
            a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
            b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
            lower = [g_operation(a[p], b[p], beta_u[p]) for p in range(len(map_u))]
            beta_l, map_l = node(lower, base + half, half)
            beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
            betas = [
                np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]])
                for p in range(len(beta_l))
            ]
            parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
            return betas, parent_map

        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = node([llr], 0, self.N)
        return sorted(zip(metrics, decisions, codewords, strict=True), key=lambda p: p[0])

    def decode(self, llr_ch):
        paths = self._scl_decode_all(llr_ch)
        if self.crc_length > 0:
            valid = []
            for pm, u_hat, _ in paths:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                pm, u_hat = valid[0]
                return u_hat, pm
        pm, u_hat, _ = paths[0]
        return u_hat, pm
