"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import sc_decode, f_operation, g_operation, _all_computed
from decoder_sc import _leftdown, _rightdown, _up, _get_up_bit


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr, bit):
    expected = 0 if llr >= 0 else 1
    if bit != expected:
        pm += abs(llr)
    return pm


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit):
    """运行 SC 直到 bit_matrix[n][target_bit] 被判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    frozen_val = 0
    position = [0, 0, n, N]

    while not (bit_matrix[n][target_bit] == 0 or bit_matrix[n][target_bit] == 1):
        layer, col, max_layer, _ = position
        span = 2 ** (max_layer - layer)
        up_llr = llr_matrix[layer][col:col + span]
        up_bit = bit_matrix[layer][col:col + span]
        half = span // 2
        left_llr = llr_matrix[layer + 1][col:col + half]
        left_bit = bit_matrix[layer + 1][col:col + half]
        right_llr = llr_matrix[layer + 1][col + half:col + span]
        right_bit = bit_matrix[layer + 1][col + half:col + span]

        if _all_computed(up_bit):
            position = _up(position)
        else:
            if _all_computed(right_bit):
                bit_matrix[layer][col:col + span] = _get_up_bit(left_bit, right_bit)
            else:
                if _all_computed(right_llr):
                    if layer == max_layer - 1:
                        right_pos = col + 1
                        if frozen_bits[right_pos]:
                            val = frozen_val
                        else:
                            val = 0 if right_llr[0] > 0 else 1
                        bit_matrix[layer + 1][col + half] = val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_computed(left_bit):
                        right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
                        llr_matrix[layer + 1][col + half:col + span] = right_llr_new
                    else:
                        if not _all_computed(left_llr):
                            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
                            llr_matrix[layer + 1][col:col + half] = left_llr_new
                        else:
                            if layer == max_layer - 1:
                                left_pos = col
                                if frozen_bits[left_pos]:
                                    val = frozen_val
                                else:
                                    val = 0 if left_llr[0] >= 0 else 1
                                bit_matrix[layer + 1][col] = val
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix, llr_matrix[n][target_bit]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        def new_path():
            llr_m = np.full((n + 1, N), np.nan)
            bit_m = np.full((n + 1, N), np.nan)
            llr_m[0] = llr_ch
            return {"llr": llr_m, "bit": bit_m, "pm": 0.0}

        paths = [new_path()]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_m = copy.deepcopy(path["llr"])
                bit_m = copy.deepcopy(path["bit"])
                llr_m, bit_m, llr_phi = _sc_step_to_bit(
                    llr_m, bit_m, self.frozen_bits, phi
                )
                decided = int(bit_m[n][phi])

                if self.frozen_bits[phi]:
                    path["llr"] = llr_m
                    path["bit"] = bit_m
                    path["pm"] = _pm_update(path["pm"], llr_phi, 0)
                    path["bit"][n][phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (decided, 1 - decided):
                        p = {
                            "llr": copy.deepcopy(llr_m),
                            "bit": copy.deepcopy(bit_m),
                            "pm": _pm_update(path["pm"], llr_phi, bit),
                        }
                        p["bit"][n][phi] = bit
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                info_bits = p["bit"][n][self.info_indices].astype(int)
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                paths = crc_ok

        best = min(paths, key=lambda p: p["pm"])
        return best["bit"][n].astype(int), best["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(1)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x).astype(float), 1.0) * 1e3
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches} (expect 0)")
    assert mismatches == 0
