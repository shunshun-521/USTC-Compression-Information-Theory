"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径仅存 u_hat 与 PM）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_idx = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def _encode_left_bits(self, offset, size, partial_u):
        if size == 1:
            return [int(partial_u[offset])]
        half = size // 2
        left = self._encode_left_bits(offset, half, partial_u)
        right = self._encode_left_bits(offset + half, half, partial_u)
        return [left[i] ^ right[i] for i in range(half)] + right

    def _llr_at_phi(self, llrs, offset, size, partial_u, phi):
        if size == 1:
            return llrs[0]
        half = size // 2
        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        if phi < offset + half:
            return self._llr_at_phi(left_llr, offset, half, partial_u, phi)
        enc_left = self._encode_left_bits(offset, half, partial_u)
        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        return self._llr_at_phi(
            right_llr, offset + half, half, partial_u, phi
        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for phi in range(self.N):
            llr_list = llr_ch.tolist()
            expanded = []
            for pm, u in paths:
                cur_llr = self._llr_at_phi(llr_list, 0, self.N, u, phi)
                if phi in self.frozen_set:
                    u2 = u.copy()
                    u2[phi] = 0
                    expanded.append((self._pm_add(pm, cur_llr, 0), u2))
                else:
                    for bit in (0, 1):
                        u2 = u.copy()
                        u2[phi] = bit
                        expanded.append((self._pm_add(pm, cur_llr, bit), u2))
            expanded.sort(key=lambda x: x[0])
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            ok_paths = []
            for pm, u in paths:
                payload = u[self.info_idx]
                if crc_check(payload, self.crc_length):
                    ok_paths.append((pm, u))
            if ok_paths:
                paths = ok_paths

        best_pm, best_u = min(paths, key=lambda x: x[0])
        return best_u.astype(int), float(best_pm)


def verify_scl_l1_equals_sc(N=64, K=32, seed=1):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(seed)
    scl = SCLDecoder(N, frozen, list_size=1)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.5)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("L=1 SCL != SC")
    return True
