"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_core(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) | fb
        if fb:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_core(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    extended = np.concatenate([bits, np.zeros(crc_length, dtype=np.int8)])
    return _crc_core(extended, poly, crc_length) == 0


class _PathState:
    __slots__ = ("beliefs", "decoded", "node_state", "pm", "node", "depth")

    def __init__(self, n, N, llr_ch):
        self.beliefs = np.zeros((n + 1, N), dtype=np.float64)
        self.decoded = np.zeros((n + 1, N), dtype=np.int8)
        self.node_state = np.zeros(2 * N - 1, dtype=np.int8)
        self.beliefs[0, :] = llr_ch
        self.pm = 0.0
        self.node = 0
        self.depth = 0

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.beliefs = self.beliefs.copy()
        p.decoded = self.decoded.copy()
        p.node_state = self.node_state.copy()
        p.pm = self.pm
        p.node = self.node
        p.depth = self.depth
        return p


class SCLDecoder:
    """SCL 译码器（与 SC 相同的树遍历，列表路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _penalty(llr, bit):
        if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0):
            return 0.0
        return float(abs(llr))

    def _step_l(self, path):
        node, depth = path.node, path.depth
        pos = (1 << depth) - 1 + node
        ci = 1 << (self.n - depth)
        inc = path.beliefs[depth, ci * node : ci * (node + 1)]
        p1, p2 = inc[: ci // 2], inc[ci // 2 :]
        node *= 2
        depth += 1
        ci //= 2
        path.beliefs[depth, ci * node : ci * (node + 1)] = f_operation(p1, p2)
        path.node_state[pos] = 1
        path.node, path.depth = node, depth

    def _step_r(self, path):
        node, depth = path.node, path.depth
        pos = (1 << depth) - 1 + node
        ci = 1 << (self.n - depth)
        inc = path.beliefs[depth, ci * node : ci * (node + 1)]
        p1, p2 = inc[: ci // 2], inc[ci // 2 :]
        ibn = 2 * node
        ld = depth + 1
        li = ci // 2
        db = path.decoded[ld, li * ibn : li * (ibn + 1)]
        node = node * 2 + 1
        depth += 1
        ci //= 2
        path.beliefs[depth, ci * node : ci * (node + 1)] = g_operation(p1, p2, db)
        path.node_state[pos] = 2
        path.node, path.depth = node, depth

    def _step_u(self, path):
        node, depth = path.node, path.depth
        ci = 1 << (self.n - depth)
        lc, rc = 2 * node, 2 * node + 1
        pd = depth + 1
        pi = ci // 2
        dl = path.decoded[pd, pi * lc : pi * (lc + 1)]
        dr = path.decoded[pd, pi * rc : pi * (rc + 1)]
        path.decoded[depth, ci * node : ci * (node + 1)] = np.concatenate(
            [(dl + dr) % 2, dr]
        )
        path.node, path.depth = node // 2, depth - 1

    def _run_path(self, path):
        done = False
        while not done:
            if path.depth == self.n:
                node = path.node
                llr = path.beliefs[self.n, node]
                yield path, node, llr
                if node == self.N - 1:
                    done = True
                else:
                    path.node //= 2
                    path.depth -= 1
                continue

            pos = (1 << path.depth) - 1 + path.node
            st = path.node_state[pos]
            if st == 0:
                self._step_l(path)
            elif st == 1:
                self._step_r(path)
            else:
                self._step_u(path)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.n, self.N, llr_ch)]

        for leaf in range(self.N):
            candidates = []
            for path in paths:
                work = path.copy()
                gen = self._run_path(work)
                _, node, llr = next(gen)
                assert node == leaf
                if leaf in self.frozen_set:
                    p = work
                    p.decoded[self.n, leaf] = 0
                    p.pm += self._penalty(llr, 0)
                    if leaf < self.N - 1:
                        p.node //= 2
                        p.depth -= 1
                    candidates.append(p)
                else:
                    for bit in (0, 1):
                        p = work.copy()
                        p.decoded[self.n, leaf] = bit
                        p.pm += self._penalty(llr, bit)
                        if leaf < self.N - 1:
                            p.node //= 2
                            p.depth -= 1
                        candidates.append(p)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        finals = []
        for path in paths:
            u_hat = path.decoded[self.n, :].astype(int)
            ok = True
            if self.crc_length > 0:
                ok = crc_check(u_hat[self.info_positions], self.crc_length)
            finals.append((path.pm, ok, u_hat))

        good = [f for f in finals if f[1]]
        if good:
            pm, _, u_hat = min(good, key=lambda x: x[0])
        else:
            pm, _, u_hat = min(finals, key=lambda x: x[0])
        return u_hat, pm
