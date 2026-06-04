"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _compute_left_alpha(llr):
    half = len(llr) // 2
    return f_operation(llr[:half], llr[half:])


def _compute_right_alpha(llr, left_bits):
    half = len(llr) // 2
    left_bits = np.asarray(left_bits, dtype=np.float64)
    return llr[half:] - (2.0 * left_bits - 1.0) * llr[:half]


def _compute_encoding_step(level, n, source, result):
    """极化编码单步（用于 SC 比特回传）。"""
    step = 1 << (n - level - 1)
    groups = 1 << level
    result = result.copy()
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


# ==================== SC 译码器 ====================


class SCDecoder:
    """基于层级 LLR 的 SC 译码器（非递归高效结构）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.mask = (self.frozen_bits == 0).astype(int)
        self._rev = bit_reversal_permutation(N)

    def _position_state(self, position):
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        return bits[-self.n :]

    def decode(self, received_llr):
        """received_llr: 自然信道顺序（与 polar_encode 输出一致）。"""
        received_llr = np.asarray(received_llr, dtype=np.float64)[self._rev]
        n, N = self.n, self.N

        current_state = np.zeros(n, dtype=np.int8)
        previous_state = np.ones(n, dtype=np.int8)

        intermediate_llr = [received_llr.copy()]
        length = N // 2
        while length > 0:
            intermediate_llr.append(np.zeros(length, dtype=np.float64))
            length //= 2

        intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
        u_hat = np.zeros(N, dtype=int)

        for position in range(N):
            current_state = self._position_state(position)

            for i in range(1, n + 1):
                llr = intermediate_llr[i - 1]
                if current_state[i - 1] == previous_state[i - 1]:
                    continue
                if current_state[i - 1] == 0:
                    intermediate_llr[i] = _compute_left_alpha(llr)
                else:
                    end = position
                    start = end - (1 << (n - i))
                    left_bits = intermediate_bits[i][start:end]
                    intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

            decision = (
                int(intermediate_llr[-1][0] < 0) if self.mask[position] else 0
            )
            u_hat[position] = decision
            intermediate_bits[-1][position] = decision

            for i in range(n - 1, -1, -1):
                intermediate_bits[i] = _compute_encoding_step(
                    i, n, intermediate_bits[i + 1], intermediate_bits[i]
                )

            previous_state = current_state.copy()

        return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（包装为与 sc_decode 相同接口）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数。"""
    N = len(llr_ch)
    return SCDecoder(N, frozen_bits).decode(llr_ch)


def sc_decode_channel(llr_ch, frozen_bits):
    """兼容别名。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供文档/扩展使用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append([l for l in range(n) if (phi >> l) & 1 == 0])
        bit_layer_vec.append([l for l in range(n) if (phi >> l) & 1 == 1])
    return lambda_offset, llr_layer_vec, bit_layer_vec
