"""
极化码 SC（串行抵消）译码器
非递归因子图 SC（与 u @ G 编码配套）+ 递归 min-sum 参考实现
"""
import numpy as np
from _sc_func import (
    all_num,
    leftdown,
    rightdown,
    up,
    get_up_bit,
    get_right_bit,
    get_right_llr,
    get_left_bit,
    get_left_llr,
)


def f_operation(La, Lb):
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_tree(y_llr, information_pos, frozen_value=0):
    y_llr = np.asarray(y_llr, dtype=np.float64)
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while all_num(bit_matrix[n]) == 0:
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = get_right_bit(
                            right_llr, information_pos, frozen_value, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if all_num(left_llr) == 0:
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(
                                    left_llr, information_pos, frozen_value, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = np.where(~frozen_bits)[0].tolist()
    return _sc_decode_tree(llr_ch, info_pos, frozen_value=0)


def sc_decode_recursive(llr, frozen_bits):
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def dec(llr_node, off):
        n = len(llr_node)
        if n == 1:
            i = off
            u_hat[i] = 0 if frozen_bits[i] else (0 if llr_node[0] >= 0 else 1)
            return
        h = n // 2
        dec(f_operation(llr_node[:h], llr_node[h:]), off)
        dec(g_operation(llr_node[:h], llr_node[h:], u_hat[off : off + h]), off + h)

    dec(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(np.log2(N))
    return [0] * (n + 1), [[] for _ in range(N)], [[] for _ in range(N)]


if __name__ == "__main__":
    from encoder import polar_encode, build_generator_matrix
    from construction import ga_construction
    from channel import compute_llr, bpsk_modulate, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, (u @ build_generator_matrix(4)) % 2)

    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + sigma * rng.standard_normal(N)
        llr = compute_llr(y, sigma)
        uh = sc_decode(llr, frozen)
        if np.array_equal(uh, u):
            ok += 1
    print(f"SC high-SNR test: {ok}/100")
