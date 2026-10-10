"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _hard_decision


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg


def _crc_to_bits(value, crc_length):
    return np.array(
        [(value >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    rem = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, _crc_to_bits(rem, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "llr_cache")

    def __init__(self, pm=0.0):
        self.pm = pm
        self.llr_cache = {}


class SCLDecoder:
    """
    SCL 译码器（树形递归 + 路径度量；L=1 时等价于 SC）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, bit):
        """不一致分支增加 |LLR| 惩罚。"""
        hard = _hard_decision(llr)
        if bit != hard:
            pm += abs(llr)
        return pm

    def _decode_tree(self, llr, frozen_slice, paths):
        """对子树做多路径译码，返回路径列表 [(u_hat_seg, u_up_seg, pm), ...]。"""
        n = len(llr)
        if n == 1:
            new_paths = []
            for path in paths:
                llr_val = llr[0]
                if frozen_slice[0]:
                    pm = self._path_metric_update(path.pm, llr_val, 0)
                    new_paths.append((np.array([0], dtype=int), np.array([0], dtype=int), pm))
                else:
                    for bit in (0, 1):
                        pm = self._path_metric_update(path.pm, llr_val, bit)
                        new_paths.append((np.array([bit], dtype=int), np.array([bit], dtype=int), pm))
            new_paths.sort(key=lambda x: x[2])
            return new_paths[: self.list_size]

        half = n // 2
        llr_upper = f_operation(llr[:half], llr[half:])
        paths_upper = self._decode_tree(llr_upper, frozen_slice[:half], paths)

        combined = []
        for u1, u1_up, pm1 in paths_upper:
            llr_lower = g_operation(llr[:half], llr[half:], u1_up)
            paths_lower = self._decode_tree(
                llr_lower, frozen_slice[half:], [_Path(pm1)]
            )
            for u2, u2_up, pm2 in paths_lower:
                u_hat = np.concatenate([u1, u2])
                u1_xor = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int))
                u_up = np.concatenate([u1_xor, u2_up.astype(int)])
                combined.append((u_hat, u_up, pm2))

        combined.sort(key=lambda x: x[2])
        return combined[: self.list_size]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._decode_tree(llr_ch, self.frozen_bits, [_Path(0.0)])
        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = []
            for u_hat, _, pm in paths:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((u_hat, pm))
            if valid:
                valid.sort(key=lambda x: x[1])
                return valid[0][0], valid[0][1]

        best = min(paths, key=lambda x: x[2])
        return best[0], best[2]
