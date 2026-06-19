"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode

CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_process(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_process(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return _crc_process(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时共享 P/C，写时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(
            N
        )
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_ch=None):
        n, N = self.n, self.N
        P = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        C = [np.zeros(N, dtype=int) for _ in range(n + 1)]
        if llr_ch is not None:
            P[n][:] = llr_ch
        return {
            "pm": 0.0,
            "P": P,
            "C": C,
            "u_hat": np.zeros(N, dtype=int),
            "P_id": [id(arr) for arr in P],
            "C_id": [id(arr) for arr in C],
        }

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "P": path["P"],
            "C": path["C"],
            "u_hat": path["u_hat"].copy(),
            "P_id": path["P_id"][:],
            "C_id": path["C_id"][:],
        }

    def _ensure_private(self, path, layer, kind):
        key = "P" if kind == "P" else "C"
        id_key = f"{key}_id"
        arrays = path[key]
        if path[id_key][layer] != id(arrays[layer]):
            arrays[layer] = arrays[layer].copy()
            path[id_key][layer] = id(arrays[layer])

    def _calc_llr(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            self._ensure_private(path, layer, "P")
            self._ensure_private(path, layer + 1, "P")
            offset = self.lambda_offset[layer]
            span = self.lambda_offset[layer + 1]
            for i in range(0, span, 2 * offset):
                for j in range(offset):
                    idx = i + j
                    La = path["P"][layer + 1][idx]
                    Lb = path["P"][layer + 1][idx + offset]
                    if (idx // offset) % 2 == 0:
                        path["P"][layer][idx] = f_operation(La, Lb)
                    else:
                        up_bit = path["C"][layer + 1][idx - offset]
                        path["P"][layer][idx] = g_operation(La, Lb, up_bit)

    def _update_bits(self, path, phi):
        path["C"][0][phi] = path["u_hat"][phi]
        for layer in self.bit_layer_vec[phi]:
            self._ensure_private(path, layer, "C")
            self._ensure_private(path, layer + 1, "C")
            offset = self.lambda_offset[layer]
            span = self.lambda_offset[layer + 1]
            for i in range(0, span, 2 * offset):
                for j in range(offset):
                    idx = i + j
                    path["C"][layer + 1][idx] = (
                        path["C"][layer][idx] ^ path["C"][layer][idx + offset]
                    )
                    path["C"][layer + 1][idx + offset] = path["C"][layer][idx + offset]

    @staticmethod
    def _path_metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._calc_llr(path, phi)
                llr = path["P"][0][phi]

                if self.frozen_bits[phi]:
                    path["pm"] += self._path_metric_penalty(llr, 0)
                    path["u_hat"][phi] = 0
                    self._update_bits(path, phi)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path["pm"] += self._path_metric_penalty(llr, bit)
                        new_path["u_hat"][phi] = bit
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
