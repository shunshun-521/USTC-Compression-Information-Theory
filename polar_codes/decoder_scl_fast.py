"""Numba 加速的 SCL 译码（可选）。"""
import numpy as np

try:
    from numba import njit

    @njit
    def _f_op(a, b):
        sa = 1.0 if a >= 0 else -1.0
        sb = 1.0 if b >= 0 else -1.0
        return sa * sb * min(abs(a), abs(b))

    @njit
    def _g_op(a, b, u):
        return (1.0 - 2.0 * u) * a + b

    @njit
    def _scl_paths(llr, frozen, list_size):
        n = llr.shape[0]
        if n == 1:
            if frozen[0]:
                pm_out = np.array([0.0])
                u_out = np.zeros((1, 1), dtype=np.int8)
                up_out = np.zeros((1, 1), dtype=np.int8)
                return pm_out, u_out, up_out
            llr_val = llr[0]
            hard = 0 if llr_val >= 0 else 1
            pm = np.empty(2)
            u_tmp = np.empty((2, 1), dtype=np.int8)
            for idx, u_bit in enumerate((0, 1)):
                pm[idx] = 0.0 if u_bit == hard else abs(llr_val)
                u_tmp[idx, 0] = u_bit
            order = np.argsort(pm)
            keep = min(list_size, 2)
            pm_out = np.empty(keep)
            u_out = np.empty((keep, 1), dtype=np.int8)
            up_out = np.empty((keep, 1), dtype=np.int8)
            for i in range(keep):
                j = order[i]
                pm_out[i] = pm[j]
                u_out[i, 0] = u_tmp[j, 0]
                up_out[i, 0] = u_tmp[j, 0]
            return pm_out, u_out, up_out

        h = n // 2
        llr_l = np.empty(h, dtype=np.float64)
        for i in range(h):
            llr_l[i] = _f_op(llr[i], llr[i + h])
        pm_l, u_l, up_l = _scl_paths(llr_l, frozen[:h], list_size)

        max_paths = pm_l.shape[0] * list_size
        pm_all = np.empty(max_paths)
        u_all = np.empty((max_paths, n), dtype=np.int8)
        up_all = np.empty((max_paths, n), dtype=np.int8)
        count = 0

        for i in range(pm_l.shape[0]):
            llr_r = np.empty(h, dtype=np.float64)
            for j in range(h):
                llr_r[j] = _g_op(llr[j], llr[j + h], up_l[i, j])
            pm_r, u_r, up_r = _scl_paths(llr_r, frozen[h:], list_size)
            for k in range(pm_r.shape[0]):
                pm_all[count] = pm_l[i] + pm_r[k]
                for j in range(h):
                    u_all[count, j] = u_l[i, j]
                    up_all[count, j] = up_l[i, j] ^ up_r[k, j]
                for j in range(h):
                    u_all[count, j + h] = u_r[k, j]
                    up_all[count, j + h] = up_r[k, j]
                count += 1

        order = np.argsort(pm_all[:count])
        keep = min(list_size, count)
        pm_out = np.empty(keep)
        u_out = np.empty((keep, n), dtype=np.int8)
        up_out = np.empty((keep, n), dtype=np.int8)
        for i in range(keep):
            j = order[i]
            pm_out[i] = pm_all[j]
            for t in range(n):
                u_out[i, t] = u_all[j, t]
                up_out[i, t] = up_all[j, t]
        return pm_out, u_out, up_out

    HAS_NUMBA_SCL = True
except ImportError:
    HAS_NUMBA_SCL = False


def scl_decode_numba(llr, frozen_bits, list_size):
    if not HAS_NUMBA_SCL:
        return None
    pm, paths, _ = _scl_paths(
        np.asarray(llr, dtype=np.float64),
        np.asarray(frozen_bits, dtype=np.bool_),
        list_size,
    )
    return pm, paths.astype(int)
