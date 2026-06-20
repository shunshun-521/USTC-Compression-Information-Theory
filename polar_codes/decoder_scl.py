"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _SCPathState, _frozen_mask_to_info_set

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(info_bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        return pm + abs(llr)
    return pm


def _is_frozen(frozen_bits, phi):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return bool(fb[phi])
    return fb[phi] == 1


def _decide_hard_bit(phi, llr, info_set, state):
    if phi not in info_set:
        return 0
    p0, p1, p2, _ = state.position
    half = 2 ** (p2 - p0 - 1)
    is_right = phi == p1 + half
    if is_right:
        return 0 if llr > 0 else 1
    return 0 if llr >= 0 else 1


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_indices = _frozen_mask_to_info_set(frozen_bits)
        self.info_set = set(self.info_indices)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"state": _SCPathState(y_llr, self.n, self.N), "pm": 0.0}]

        while not all(p["state"].is_finished() for p in paths):
            new_paths = []
            for path in paths:
                state = path["state"]
                if state.is_finished():
                    new_paths.append(path)
                    continue

                pending = state.pending_decision()
                if pending is None:
                    state.step()
                    new_paths.append(path)
                    continue

                phi, llr = pending
                if _is_frozen(self.frozen_bits, phi):
                    bit_val = 0
                    child_state = state.copy()
                    child_state.apply_bit(phi, bit_val)
                    new_paths.append(
                        {
                            "state": child_state,
                            "pm": _path_metric_update(path["pm"], llr, bit_val),
                        }
                    )
                else:
                    for bit_val in (0, 1):
                        child_state = state.copy()
                        child_state.apply_bit(phi, bit_val)
                        new_paths.append(
                            {
                                "state": child_state,
                                "pm": _path_metric_update(path["pm"], llr, bit_val),
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        candidates = []
        for path in paths:
            u_hat = path["state"].bit_matrix[self.n].astype(int)
            candidates.append((path["pm"], u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                candidates = valid

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat.copy(), pm
