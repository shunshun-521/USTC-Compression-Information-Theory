"""工具函数：结果保存、绘图、容量计算"""
import csv
import os
import struct
import zlib

import numpy as np


def save_results_csv(results, filepath):
    """将仿真结果保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = [
        "eb_n0_db",
        "bler",
        "ber",
        "num_errors",
        "num_frames",
        "avg_decode_time_ms",
        "avg_iters",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "eb_n0_db": row["eb_n0_db"],
                "bler": row["bler"],
                "ber": row["ber"],
                "num_errors": row["num_errors"],
                "num_frames": row["num_frames"],
                "avg_decode_time_ms": row["avg_decode_time"] * 1000.0,
                "avg_iters": "" if row["avg_iters"] is None else row["avg_iters"],
            })


def load_results_csv(filepath):
    """从 CSV 文件加载仿真结果，返回 dict 列表。"""
    results = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_iters = row.get("avg_iters", "")
            results.append({
                "eb_n0_db": float(row["eb_n0_db"]),
                "bler": float(row["bler"]),
                "ber": float(row["ber"]),
                "num_errors": int(row["num_errors"]),
                "num_frames": int(row["num_frames"]),
                "avg_decode_time": float(row["avg_decode_time_ms"]) / 1000.0,
                "avg_iters": None if avg_iters == "" else float(avg_iters),
            })
    return results


def compute_bpsk_capacity(eb_n0_db_list, rate):
    """
    计算 BPSK 离散输入信道容量（bits/channel use）。
    """
    eb_n0_db_list = np.atleast_1d(eb_n0_db_list)
    capacities = []
    y = np.linspace(-10, 10, 10001)
    dy = y[1] - y[0]

    for eb_n0_db in eb_n0_db_list:
        ebn0_lin = 10.0 ** (eb_n0_db / 10.0)
        sigma = 1.0 / np.sqrt(2.0 * rate * ebn0_lin)
        p0 = np.exp(-0.5 * (y - 1) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
        p1 = np.exp(-0.5 * (y + 1) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
        py = 0.5 * (p0 + p1)
        llr = 2.0 * y / (sigma ** 2)
        p0_post = 1.0 / (1.0 + np.exp(-llr))
        p1_post = 1.0 - p0_post
        h = -(p0_post * np.log2(p0_post + 1e-300) + p1_post * np.log2(p1_post + 1e-300))
        c = 1.0 - np.sum(h * py) * dy
        capacities.append(float(c))

    return np.array(capacities)


def find_capacity_limit(rate, eb_n0_range=(-2, 10), num_points=1000):
    """找到使 BPSK 信道容量等于码率 R 的 Eb/N0（dB）。"""
    lo, hi = eb_n0_range

    cap_lo = compute_bpsk_capacity(lo, rate)[0]
    cap_hi = compute_bpsk_capacity(hi, rate)[0]
    if cap_lo > rate:
        return float(lo)
    if cap_hi < rate:
        return float(hi)

    for _ in range(80):
        mid = (lo + hi) / 2.0
        cap = compute_bpsk_capacity(mid, rate)[0]
        if cap > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _png_write(path, pixels, width, height):
    """将 RGB 像素数组写入 PNG（无第三方依赖）。"""
    rows = []
    for row in range(height):
        row_bytes = bytearray([0])
        for col in range(width):
            r, g, b = pixels[row * width + col]
            row_bytes.extend([r, g, b])
        rows.append(bytes(row_bytes))
    raw = b"".join(rows)
    compressed = zlib.compress(raw, 9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def _draw_semilogy_png(save_path, results_dict, title, shannon_limit_db=None):
    """纯 NumPy 绘制 semilogy BLER 曲线并保存 PNG。"""
    width, height = 960, 600
    margin = 70
    pixels = [(255, 255, 255)] * (width * height)

    def set_px(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pixels[y * width + x] = color

    def draw_line(x0, y0, x1, y1, color):
        steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            set_px(x, y, color)

    all_eb = []
    all_bler = []
    for results in results_dict.values():
        all_eb.extend([r["eb_n0_db"] for r in results])
        all_bler.extend([max(r["bler"], 1e-6) for r in results])

    x_min, x_max = min(all_eb), max(all_eb)
    y_min, y_max = 1e-4, 1.0

    def x_map(x):
        return margin + int((x - x_min) / max(x_max - x_min, 1e-9) * (width - 2 * margin))

    def y_map(y):
        y = np.clip(y, y_min, y_max)
        logy = (np.log10(y) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min))
        return height - margin - int(logy * (height - 2 * margin))

    axis_color = (0, 0, 0)
    draw_line(margin, height - margin, width - margin, height - margin, axis_color)
    draw_line(margin, margin, margin, height - margin, axis_color)

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]

    for ci, (label, results) in enumerate(results_dict.items()):
        color = colors[ci % len(colors)]
        pts = [(x_map(r["eb_n0_db"]), y_map(max(r["bler"], 1e-6))) for r in results]
        for i in range(len(pts) - 1):
            draw_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color)

    if shannon_limit_db is not None:
        xv = x_map(shannon_limit_db)
        draw_line(xv, margin, xv, height - margin, (160, 160, 160))

    _png_write(save_path, pixels, width, height)

    svg_path = os.path.splitext(save_path)[0] + ".svg"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write(f'<rect width="100%" height="100%" fill="white"/>\n')
        f.write(f'<text x="{margin}" y="30" font-size="16">{title}</text>\n')
        f.write(
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>\n'
        )
        f.write(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>\n')
        for ci, (label, results) in enumerate(results_dict.items()):
            color = colors[ci % len(colors)]
            hex_color = "#%02x%02x%02x" % color
            pts = " ".join(
                f"{x_map(r['eb_n0_db'])},{y_map(max(r['bler'], 1e-6))}" for r in results
            )
            f.write(f'<polyline fill="none" stroke="{hex_color}" stroke-width="2" points="{pts}"/>\n')
            f.write(f'<text x="{width - 200}" y="{40 + ci * 18}" fill="{hex_color}" font-size="12">{label}</text>\n')
        if shannon_limit_db is not None:
            xv = x_map(shannon_limit_db)
            f.write(f'<line x1="{xv}" y1="{margin}" x2="{xv}" y2="{height - margin}" stroke="#999" stroke-dasharray="4"/>\n')
        f.write("</svg>\n")


def plot_bler_curves(results_dict, title, save_path, shannon_limit_db=None,
                     xlabel='Eb/N0 (dB)', ylabel='BLER'):
    """绘制 BLER-Eb/N0 曲线（优先 matplotlib，否则纯 Python PNG/SVG）。"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, results in results_dict.items():
            eb = [r["eb_n0_db"] for r in results]
            bler = [max(r["bler"], 1e-6) for r in results]
            ax.semilogy(eb, bler, "o-", label=label, markersize=4)

        if shannon_limit_db is not None:
            ax.axvline(shannon_limit_db, color="gray", linestyle="--", label="Shannon limit")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path)
        plt.close()
    except ImportError:
        _draw_semilogy_png(save_path, results_dict, title, shannon_limit_db)


def save_frozen_set_info(N_list, K, design_eb_n0_db, save_path):
    """将各码长的信息位/冻结位集合保存到文本文件。"""
    from construction import ga_construction

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for N in N_list:
            k_val = N // 2 if K is None else K
            rate = k_val / N
            info_idx, frozen_idx, _ = ga_construction(N, k_val, design_eb_n0_db)
            f.write("=" * 53 + "\n")
            f.write(
                f"N={N}, K={k_val}, design_Eb/N0={design_eb_n0_db} dB, R={rate:.4f}\n"
            )
            f.write("=" * 53 + "\n")
            f.write(f"Info indices (all {len(info_idx)}):\n")
            f.write(np.array2string(info_idx, threshold=N, max_line_width=120) + "\n")
            f.write(f"Frozen indices (all {len(frozen_idx)}):\n")
            f.write(np.array2string(frozen_idx, threshold=N, max_line_width=120) + "\n")
            f.write("-" * 53 + "\n")
