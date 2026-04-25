"""
分块跑 GPTzip-medium 大文件版
用法: python run_medium_chunked.py 100KB
"""
import os, sys, hashlib, subprocess, shutil, time, struct, csv

PY = sys.executable
SCRIPT = "GPTzip/gptzip.py"
SLICE = sys.argv[1] if len(sys.argv) > 1 else "100KB"
LABEL = "GPTzip-medium"
CHUNK = 25 * 1024   # 25KB per chunk

src = f"data/enwik8_{SLICE}.txt"
out_dir = f"data/{LABEL}_{SLICE}_chunks"
os.makedirs(out_dir, exist_ok=True)

raw = open(src, "rb").read()
print(f"原始 {len(raw)} B, 切成 {CHUNK} B 一块")

# 切片
chunk_paths = []
for i in range(0, len(raw), CHUNK):
    p = f"{out_dir}/chunk_{i//CHUNK:03d}.txt"
    open(p, "wb").write(raw[i:i+CHUNK])
    chunk_paths.append(p)
print(f"共 {len(chunk_paths)} 块")

# 逐块压缩
total_comp = 0
t_total = 0
for p in chunk_paths:
    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-z", p, "-o", p.replace(".txt",".gpz")], check=True)
    t_total += time.perf_counter() - t0
    total_comp += os.path.getsize(p.replace(".txt",".gpz.gpz"))

# 逐块解压
dec_total = b""
t_d = 0
for p in chunk_paths:
    gpz = p.replace(".txt",".gpz.gpz")
    dec = p.replace(".txt",".dec.txt")
    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-u", gpz, "-o", dec], check=True)
    t_d += time.perf_counter() - t0
    dec_total += open(dec, "rb").read()

ok = hashlib.md5(raw).hexdigest() == hashlib.md5(dec_total[:len(raw)]).hexdigest()
bpb = total_comp * 8 / len(raw)
thr = (len(raw)/1e6) / max(t_total, 1e-6)

row = [LABEL, SLICE, len(raw), total_comp, round(bpb,4),
       round(t_total,4), round(t_d,4), round(thr,6),
       "通过(分块)" if ok else "失败"]
print("\n结果:", row)

# 追加到 csv
with open("results_GPTzip-medium.csv", "a", newline="", encoding="utf-8-sig") as f:
    csv.writer(f).writerow(row)
print("已追加到 results_GPTzip-medium.csv")
