"""SC/SCL 译码核心（改编自公开参考实现）。"""
import numpy as np

try:
    import numba

    @numba.jit(nopython=True)
    def _f_node_minsum(a, b):
        return np.sign(a * b) * np.minimum(np.abs(a), np.abs(b))

    @numba.jit(nopython=True)
    def _g_node(llr1, llr2, bit):
        return llr1 * (1 - 2 * bit) + llr2

    @numba.jit(nopython=True)
    def _b_check(level, idx):
        return (idx // (1 << level)) % 2

    @numba.jit(nopython=True)
    def _s_updater(level, idx, s):
        if _b_check(level - 1, idx):
            s[level, idx] = s[level - 1, idx]
        else:
            if s[level - 1, idx] == -1:
                _s_updater(level - 1, idx, s)
            sibling = idx + (1 << (level - 1))
            if s[level - 1, sibling] == -1:
                _s_updater(level - 1, sibling, s)
            s[level, idx] = s[level - 1, idx] ^ s[level - 1, sibling]

    @numba.jit(nopython=True)
    def _li(level, idx, llrs, s):
        if llrs[level, idx] != -np.inf:
            return llrs[level, idx]
        if _b_check(level, idx) == 0:
            llrs[level, idx] = _f_node_minsum(
                _li(level + 1, idx, llrs, s),
                _li(level + 1, idx + (1 << level), llrs, s),
            )
        else:
            if level > 0:
                _s_updater(level, idx - (1 << level), s)
            llrs[level, idx] = _g_node(
                _li(level + 1, idx - (1 << level), llrs, s),
                _li(level + 1, idx, llrs, s),
                s[level, idx - (1 << level)],
            )
        return llrs[level, idx]

    @numba.jit(nopython=True)
    def sc_decode_core(llr_channel, is_info_bit):
        n = int(np.log2(len(llr_channel)))
        n_rows = n + 1
        n_cols = 1 << n
        llrs = -np.inf * np.ones((n_rows, n_cols), dtype=np.float32)
        s = -1 * np.ones((n_rows, n_cols), dtype=np.int8)
        llrs[n, :] = llr_channel.astype(np.float32)
        decoded = np.zeros(n_cols, dtype=np.int8)
        for idx in range(n_cols):
            if is_info_bit[idx] == 0:
                s[0, idx] = 0
                llrs[0, idx] = np.inf
            else:
                llrs[0, idx] = _li(0, idx, llrs, s)
                s[0, idx] = 1 if llrs[0, idx] < 0 else 0
            decoded[idx] = s[0, idx]
        return decoded

    @numba.jit(nopython=True)
    def scl_decode_core(llr_channel, is_info_bit, list_size):
        n = int(np.log2(len(llr_channel)))
        n_rows = n + 1
        n_cols = 1 << n
        llrs_list = []
        s_list = []
        for _ in range(list_size):
            llrs = -np.inf * np.ones((n_rows, n_cols), dtype=np.float32)
            s = -1 * np.ones((n_rows, n_cols), dtype=np.int8)
            llrs[n, :] = llr_channel.astype(np.float32)
            llrs_list.append(llrs)
            s_list.append(s)

        pm = np.full(list_size, np.float32(np.inf), dtype=np.float32)
        pm[0] = np.float32(0.0)
        pm_dm = np.zeros(2 * list_size, dtype=np.float32)
        dm = np.zeros(list_size, dtype=np.float32)

        for idx in range(n_cols):
            for dd in range(list_size):
                if is_info_bit[idx] == 0:
                    s_list[dd][0, idx] = 0
                    llrs_list[dd][0, idx] = np.float32(np.inf)
                else:
                    llrs_list[dd][0, idx] = _li(0, idx, llrs_list[dd], s_list[dd])
                    s_list[dd][0, idx] = 1 if llrs_list[dd][0, idx] < 0 else 0
                    dm[dd] = np.float32(np.abs(llrs_list[dd][0, idx]))

            if is_info_bit[idx] != 0 and list_size > 1:
                for dd in range(list_size):
                    pm_dm[dd] = pm[dd]
                    pm_dm[list_size + dd] = np.float32(pm[dd] + dm[dd])
                idx_sort = np.argsort(pm_dm)
                new_llrs = []
                new_s = []
                new_pm = np.zeros(list_size, dtype=np.float32)
                for pos in range(list_size):
                    chosen = idx_sort[pos]
                    if chosen < list_size:
                        new_llrs.append(llrs_list[chosen])
                        new_s.append(s_list[chosen])
                        new_pm[pos] = pm[chosen]
                    else:
                        src = chosen - list_size
                        llrs_copy = llrs_list[src].copy()
                        s_copy = s_list[src].copy()
                        s_copy[0, idx] = 1 - s_list[src][0, idx]
                        new_llrs.append(llrs_copy)
                        new_s.append(s_copy)
                        new_pm[pos] = pm_dm[chosen]
                llrs_list = new_llrs
                s_list = new_s
                pm = new_pm

        best = int(np.argmin(pm))
        return s_list[best][0, :].astype(np.int8), pm[best]

except ImportError:
    numba = None
