"""
极化码 SC（串行抵消）译码器
核心 SC 树遍历改编自公开教学实现（与 G=F^{⊗n} 编码配套）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        s1 = 1 if La == 0 else np.sign(La)
        s2 = 1 if Lb == 0 else np.sign(Lb)
        return s1 * s2 * min(abs(La), abs(Lb))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_num(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))) * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_bit(right_llr, information_pos, frozen_val, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_val


def _get_left_bit(left_llr, information_pos, frozen_val, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _sc_tree_decode(y_llr, information_pos, frozen_val=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = np.ones((n + 1, N), dtype=np.float64)
    bit_matrix[:] = np.nan
    llr_matrix[0] = y_llr
    information_pos = set(information_pos)
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = _get_right_bit(right_llr, information_pos, frozen_val, right_bit_pos)
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not _all_num(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = _get_left_bit(left_llr, information_pos, frozen_val, left_bit_pos)
                                bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def _align_llr_to_generator(llr_ch, bit_reverse_codeword):
    """编码含 B_N 倒序时，将信道 LLR 对齐到 G=F^{⊗n} 的自然顺序"""
    N = len(llr_ch)
    if not bit_reverse_codeword:
        return llr_ch
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return llr_ch[inv]


def sc_decode(llr_ch, frozen_bits, bit_reverse_codeword=True, info_indices=None):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    if info_indices is None:
        info_indices = np.where(~frozen_bits)[0]
    llr = _align_llr_to_generator(np.asarray(llr_ch, dtype=np.float64), bit_reverse_codeword)
    return _sc_tree_decode(llr, info_indices, frozen_val=0)


def sc_decode_recursive(llr_ch, frozen_bits, **kwargs):
    return sc_decode(llr_ch, frozen_bits, **kwargs)


def validate_sc(N=64, K=32, num_frames=100, eb_n0_db=10.0, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = 1e-6  # 无损校验
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits, info_indices=info_idx)
        if not np.array_equal(u_hat[info_idx], payload):
            raise AssertionError("SC decode mismatch")
    return True
