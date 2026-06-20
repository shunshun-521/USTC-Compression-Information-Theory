"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_mask_to_info_set(frozen_bits):
    """将 frozen_bits（1=冻结，0=信息）转为信息位索引列表。"""
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return np.where(~frozen_bits)[0].tolist()
    return np.where(frozen_bits == 0)[0].tolist()


def _is_filled(arr):
    return not np.isnan(arr).any()


def get_right_llr(left_bit, up_llr):
    """根据左子树比特计算右子树 LLR。"""
    half = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
    )


class _SCPathState:
    """SC/SCL 共享的树遍历状态。"""

    __slots__ = ("llr_matrix", "bit_matrix", "position", "n", "N")

    def __init__(self, y_llr, n, N):
        self.n = n
        self.N = N
        self.llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        self.bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        self.llr_matrix[0] = np.asarray(y_llr, dtype=np.float64)
        self.position = [0, 0, n, N]

    def copy(self):
        other = _SCPathState(self.llr_matrix[0], self.n, self.N)
        other.llr_matrix = self.llr_matrix.copy()
        other.bit_matrix = self.bit_matrix.copy()
        other.position = self.position.copy()
        return other

    def _up(self):
        p0, p1, p2, p3 = self.position
        p1 = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
        self.position = [p0 - 1, p1, p2, p3]

    def _leftdown(self):
        p0, p1, p2, p3 = self.position
        self.position = [p0 + 1, p1, p2, p3]

    def _rightdown(self):
        p0, p1, p2, p3 = self.position
        self.position = [p0 + 1, p1 + 2 ** (p2 - p0 - 1), p2, p3]

    def is_finished(self):
        return _is_filled(self.bit_matrix[self.n])

    def pending_decision(self):
        """若下一比特待判决，返回 (phi, llr)；否则返回 None。"""
        p0, p1, p2, _ = self.position
        if p0 != p2 - 1:
            return None
        span = 2 ** (p2 - p0)
        half = span // 2
        left_llr = self.llr_matrix[p0 + 1][p1 : p1 + half]
        right_llr = self.llr_matrix[p0 + 1][p1 + half : p1 + span]
        if not _is_filled(self.bit_matrix[p0 + 1][p1 : p1 + half]) and _is_filled(left_llr):
            return p1, left_llr[0]
        if (
            not _is_filled(self.bit_matrix[p0 + 1][p1 + half : p1 + span])
            and _is_filled(right_llr)
        ):
            return p1 + 1, right_llr[0]
        return None

    def step(self):
        """执行单步树遍历。若完成比特判决，返回 (phi, llr)。"""
        if self.is_finished():
            return None

        position = self.position
        p0, p1, p2, _ = position
        span = 2 ** (p2 - p0)
        half = span // 2

        up_llr = self.llr_matrix[p0][p1 : p1 + span]
        up_bit = self.bit_matrix[p0][p1 : p1 + span]
        left_llr = self.llr_matrix[p0 + 1][p1 : p1 + half]
        left_bit = self.bit_matrix[p0 + 1][p1 : p1 + half]
        right_llr = self.llr_matrix[p0 + 1][p1 + half : p1 + span]
        right_bit = self.bit_matrix[p0 + 1][p1 + half : p1 + span]

        if _is_filled(up_bit):
            self._up()
            return None

        if _is_filled(right_bit):
            combined = np.vstack([(left_bit + right_bit) % 2, right_bit]).reshape(1, -1)
            self.bit_matrix[p0][p1 : p1 + span] = combined
            return None

        if _is_filled(right_llr):
            if position[0] == position[2] - 1:
                return p1 + 1, right_llr[0]
            self._rightdown()
            return None

        if _is_filled(left_bit):
            self.llr_matrix[p0 + 1][p1 + half : p1 + span] = get_right_llr(
                left_bit, up_llr
            )
            return None

        if not _is_filled(left_llr):
            self.llr_matrix[p0 + 1][p1 : p1 + half] = f_operation(
                up_llr[:half], up_llr[half:]
            )
            return None

        if position[0] == position[2] - 1:
            return p1, left_llr[0]
        self._leftdown()
        return None

    def apply_bit(self, phi, bit_val):
        """在叶子节点写入判决比特并向上回溯。"""
        p0, p1, p2, _ = self.position
        span = 2 ** (p2 - p0)
        half = span // 2
        if phi == p1:
            self.bit_matrix[p0 + 1][p1] = bit_val
        else:
            self.bit_matrix[p0 + 1][p1 + half] = bit_val
        self._up()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托给非递归树遍历实现）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * N
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        layers_llr = []
        temp = phi
        for l in range(n):
            if (temp & 1) == 0:
                layers_llr.append(l)
                break
            layers_llr.append(l)
            temp >>= 1
        llr_layer_vec[phi] = layers_llr

        layers_bit = [l for l in range(n) if (phi >> l) & 1]
        bit_layer_vec[phi] = layers_bit
        lambda_offset[phi] = sum(1 << l for l in layers_llr)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1/True 表示冻结位，0/False 表示信息位。
    """
    info_indices = _frozen_mask_to_info_set(frozen_bits)
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = y_llr.size
    n = int(math.log2(N))
    info_set = set(info_indices)
    state = _SCPathState(y_llr, n, N)

    while not state.is_finished():
        pending = state.pending_decision()
        if pending is not None:
            phi, llr = pending
            if phi in info_set:
                p0, p1, p2, _ = state.position
                half = 2 ** (p2 - p0 - 1)
                is_right = phi == p1 + half
                if is_right:
                    bit_val = 0 if llr > 0 else 1
                else:
                    bit_val = 0 if llr >= 0 else 1
            else:
                bit_val = 0
            state.apply_bit(phi, bit_val)
            continue
        state.step()

    return state.bit_matrix[n].astype(int)
