"""Qwen 批量脚本: python run_qwen.py 1KB 10KB 100KB"""
import os, sys, hashlib, subprocess, time, csv

PY = sys.executable
SLICES = sys.argv[1:] if len(sys.argv) > 1 else ["1KB", "10KB", "100KB"]
LABEL = "Qwen2.5-0.5B"
rows = []

for SLICE in SLICES:
    print(f"\n========== {LABEL} on {SLICE} ==========")
    src = f"data/enwik8_{SLICE}.txt"
    qwz = f"data/{LABEL}_{SLICE}.qwz"
    dec = f"data/{LABEL}_{SLICE}.dec.txt"
    for f in [qwz, dec]:
        if os.path.exists(f): os.remove(f)

    t0 = time.perf_counter()
    subprocess.run([PY, "qwen_zip.py", "-z", src, qwz], check=True)
    t_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    subprocess.run([PY, "qwen_zip.py", "-u", qwz, dec], check=True)
    t_d = time.perf_counter() - t0

    raw_b = open(src, "rb").read()
    dec_b = open(dec, "rb").read()
    ok = hashlib.md5(raw_b).hexdigest() == hashlib.md5(dec_b).hexdigest()
    comp = os.path.getsize(qwz)
    bpb  = comp * 8 / len(raw_b)
    thr  = (len(raw_b)/1e6) / max(t_c, 1e-6)
    row = [LABEL, SLICE, len(raw_b), comp, round(bpb,4),
           round(t_c,4), round(t_d,4), round(thr,6),
           "通过" if ok else "失败"]
    print(row); rows.append(row)

with open("results_qwen.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerows(rows)
print("\n已写入 results_qwen.csv")
