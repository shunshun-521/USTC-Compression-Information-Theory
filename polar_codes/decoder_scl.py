"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, precompute_sc_indices, _map_u_from_tree


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("pm", "u_hat", "P", "C", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.P = np.zeros((n + 1, 2 * N), dtype=np.float64)
        self.C = np.zeros((n + 1, 2 * N), dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径列表 + 按需复制 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.rev = bit_reversal_permutation(N)
        self.frozen_tree = self.frozen_bits[self.rev].astype(int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _copy_path_state(self, src, dst):
        dst.P[:] = src.P
        dst.C[:] = src.C
        dst.u_hat[:] = src.u_hat
        dst.pm = src.pm

    def _update_llr_layers(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            lam = self.lambda_offset[layer]
            for beta in range(0, self.N, 2 * lam):
                for omega in range(lam):
                    idx = beta + omega
                    path.P[layer, idx] = f_operation(
                        path.P[layer + 1, idx], path.P[layer + 1, idx + lam]
                    )
                    path.P[layer, idx + lam] = g_operation(
                        path.P[layer + 1, idx],
                        path.P[layer + 1, idx + lam],
                        path.C[layer, idx],
                    )

    def _bit_backprop(self, path, phi):
        path.C[0, 0] = path.u_hat[phi]
        for layer in self.bit_layer_vec[phi]:
            lam = self.lambda_offset[layer]
            for beta in range(0, self.N, 2 * lam):
                for omega in range(lam):
                    idx = beta + omega
                    path.C[layer + 1, idx] = (path.C[layer, idx] + path.C[layer, idx + lam]) % 2
                    path.C[layer + 1, idx + lam] = path.C[layer, idx + lam]

    def _pm_penalty(self, llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [_Path(self.n, self.N)]
        paths[0].P[self.n, : self.N] = llr_ch

        for phi in range(self.N):
            candidates = []
            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._copy_path_state(path, path)
                self._update_llr_layers(path, phi)
                llr0 = path.P[0, 0]

                if self.frozen_tree[phi]:
                    pen = self._pm_penalty(llr0, 0)
                    path.pm += pen
                    path.u_hat[phi] = 0
                    self._bit_backprop(path, phi)
                    candidates.append((path.pm, pidx, None))
                else:
                    for u in (0, 1):
                        new_path = _Path(self.n, self.N)
                        self._copy_path_state(path, new_path)
                        new_path.pm += self._pm_penalty(llr0, u)
                        new_path.u_hat[phi] = u
                        self._bit_backprop(new_path, phi)
                        candidates.append((new_path.pm, -1, new_path))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, pidx, new_p in candidates:
                if len(new_paths) >= self.list_size:
                    break
                if new_p is None:
                    paths[pidx].pm = pm
                    new_paths.append(paths[pidx])
                else:
                    new_p.pm = pm
                    new_paths.append(new_p)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_orig = _map_u_from_tree(p.u_hat, self.N)
                if crc_check(u_orig[self.info_indices], self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p.pm)
        return _map_u_from_tree(best.u_hat.copy(), self.N), best.pm


def verify_scl_equals_sc(N=64, frozen_bits=None, seed=1):
    """L=1 SCL 应与 SC 一致。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(3.0, 0.5)
    rng = np.random.default_rng(seed)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
