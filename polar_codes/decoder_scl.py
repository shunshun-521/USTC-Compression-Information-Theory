"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import polar_sc_core as sc_core


def _crc_polynomial(crc_n):
    if crc_n == 8:
        loc = [8, 2, 1, 0]
    elif crc_n == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("crc_length must be 8 or 16")
    poly = [0] * (crc_n + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """CRC 编码，与 PolarCodesPython 参考实现一致。"""
    info = [int(b) for b in info_bits]
    poly = _crc_polynomial(crc_length)
    work = info + [0] * crc_length
    times = len(info)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= poly[j]
    check_code = work[-crc_length:]
    return np.array(info + check_code, dtype=np.int32)


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    if crc_length == 0:
        return True
    bits = [int(b) for b in bits]
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return all(bits[i] == expected[i] for i in range(len(bits)))


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码至 split_pos 位判决完成。"""
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    loc = sc_core.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        s, e = position[1], position[1] + span
        mid = s + span // 2
        up_llr = llr_matrix[position[0], s:e]
        up_bit = bit_matrix[position[0], s:e]
        left_llr = llr_matrix[position[0] + 1, s:mid]
        left_bit = bit_matrix[position[0] + 1, s:mid]
        right_llr = llr_matrix[position[0] + 1, mid:e]
        right_bit = bit_matrix[position[0] + 1, mid:e]

        if sc_core.all_num(up_bit):
            position = sc_core.up(position)
        elif sc_core.all_num(right_bit):
            bit_matrix[position[0], s:e] = sc_core.get_up_bit(left_bit, right_bit)
        elif sc_core.all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = sc_core.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1, mid:e] = rb
            else:
                position = sc_core.rightdown(position)
        elif sc_core.all_num(left_bit):
            llr_matrix[position[0] + 1, mid:e] = sc_core.get_right_llr(left_bit, up_llr)
        elif sc_core.all_num(left_llr) == 0:
            llr_matrix[position[0] + 1, s:mid] = sc_core.get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            lb = sc_core.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
            bit_matrix[position[0] + 1, s:mid] = lb
        else:
            position = sc_core.leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode_core(y_llr, information_pos, frozen_bit, list_size, crc_length):
    """SCL 译码核心（参考 PolarCodesPython）。"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0, :] = y_llr

    split_pos = list(information_pos)
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1

    while split_len - 1 >= split_loc:
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]

            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, split_pos[split_loc]
            )
            llr_list[i] = llr_out
            bit_list[i] = bit_out

            prev = 0 if split_loc == 0 else split_pos[split_loc - 1] + 1
            curr = split_pos[split_loc] + 1
            right_pm = sc_core.get_pm_update(llr_out[n, prev:curr], bit_out[n, prev:curr], "hf")
            pm_list[i] = pm_temp + right_pm

            llr_list.append(llr_out.copy())
            bit_wrong = bit_out.copy()
            bit_wrong[n, split_pos[split_loc]] = 1 - bit_wrong[n, split_pos[split_loc]]
            bit_list.append(bit_wrong)
            wrong_pm = sc_core.get_pm_update(llr_out[n, prev:curr], bit_wrong[n, prev:curr], "hf")
            pm_list.append(pm_temp + wrong_pm)

        if l_now > list_size / 2:
            keep = np.argsort(pm_list)[:list_size]
            pm_list = [pm_list[i] for i in keep]
            llr_list = [llr_list[i] for i in keep]
            bit_list = [bit_list[i] for i in keep]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_temp = llr_list[i]
            bit_temp = bit_list[i]
            pm_temp = pm_list[i]
            llr_out, bit_out = _sc_stepping_decoder(
                llr_temp.copy(), bit_temp.copy(), information_pos, frozen_bit, N - 1
            )
            llr_list[i] = llr_out
            bit_list[i] = bit_out
            prev = split_pos[split_loc - 1] + 1
            tail_pm = sc_core.get_pm_update(llr_out[n, prev:N], bit_out[n, prev:N], "hf")
            pm_list[i] = pm_temp + tail_pm

    pm_argsort = np.argsort(pm_list)
    best_u = None
    best_pm = pm_list[pm_argsort[0]]

    for idx in pm_argsort:
        u_temp = bit_list[idx][n]
        u_hat = np.array([0 if u_temp[i] == 0 else 1 for i in range(N)], dtype=np.int32)
        if crc_length > 0:
            info_bits = u_hat[information_pos]
            if crc_check(info_bits, crc_length):
                return u_hat, pm_list[idx]
        elif best_u is None:
            best_u = u_hat
            best_pm = pm_list[idx]

    if best_u is None:
        u_temp = bit_list[pm_argsort[0]][n]
        best_u = np.array([0 if u_temp[i] == 0 else 1 for i in range(N)], dtype=np.int32)
        best_pm = pm_list[pm_argsort[0]]

    return best_u, best_pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = list(np.where(~self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat, pm = _scl_decode_core(
            llr_ch,
            self.information_pos,
            0,
            self.list_size,
            self.crc_length,
        )
        return u_hat, pm


def verify_scl_equals_sc(N=64, K=32, num_frames=50, eb_n0_db=10.0):
    """L=1 时 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(123)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed.")
