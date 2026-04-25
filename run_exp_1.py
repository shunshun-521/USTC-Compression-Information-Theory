import os, time, gzip, bz2, lzma, hashlib, csv
import zstandard as zstd
import pathlib

pathlib.Path("data").mkdir(exist_ok=True)

# ---------- 切片 ----------
SIZES = [("1KB", 1024), ("10KB", 10240), ("100KB", 102400), ("1MB", 1048576)]
with open("enwik8", "rb") as f:
    full = f.read(1048576)  # 只读前 1MB 就够了
for label, n in SIZES:
    with open(f"data/enwik8_{label}.bin", "wb") as f:
        f.write(full[:n])

# ---------- 四种压缩器 ----------
def md5(b): return hashlib.md5(b).hexdigest()

def run_gzip(data):
    t0 = time.perf_counter(); comp = gzip.compress(data, compresslevel=9); tc = time.perf_counter()-t0
    t0 = time.perf_counter(); dec  = gzip.decompress(comp);                td = time.perf_counter()-t0
    return comp, dec, tc, td

def run_bz2(data):
    t0 = time.perf_counter(); comp = bz2.compress(data, compresslevel=9);  tc = time.perf_counter()-t0
    t0 = time.perf_counter(); dec  = bz2.decompress(comp);                  td = time.perf_counter()-t0
    return comp, dec, tc, td

def run_xz(data):  # lzma 就是 xz 用的算法
    t0 = time.perf_counter(); comp = lzma.compress(data, preset=9);         tc = time.perf_counter()-t0
    t0 = time.perf_counter(); dec  = lzma.decompress(comp);                  td = time.perf_counter()-t0
    return comp, dec, tc, td

def run_zstd(data):
    cctx = zstd.ZstdCompressor(level=22)
    dctx = zstd.ZstdDecompressor()
    t0 = time.perf_counter(); comp = cctx.compress(data);                    tc = time.perf_counter()-t0
    t0 = time.perf_counter(); dec  = dctx.decompress(comp);                  td = time.perf_counter()-t0
    return comp, dec, tc, td

METHODS = [("gzip", run_gzip), ("bzip2", run_bz2), ("xz", run_xz), ("zstd", run_zstd)]

# ---------- 跑实验 ----------
rows = []
for label, _ in SIZES:
    with open(f"data/enwik8_{label}.bin", "rb") as f:
        raw = f.read()
    for name, fn in METHODS:
        comp, dec, tc, td = fn(raw)
        ok  = (md5(raw) == md5(dec))     # 无损校验
        bpb = len(comp) * 8 / len(raw)
        thr = (len(raw) / 1e6) / max(tc, 1e-6)  # MB/s
        row = [name, label, len(raw), len(comp), round(bpb,4),
               round(tc,4), round(td,4), round(thr,2), "通过" if ok else "失败"]
        rows.append(row); print(row)

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerows(rows)
print("结果已写入 results.csv")