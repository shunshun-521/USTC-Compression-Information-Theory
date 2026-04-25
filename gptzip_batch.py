import os, sys, hashlib, subprocess, shutil, time, csv

ROOT  = r"C:\Users\28624\Informationtheory\Compression"
PY    = sys.executable
SCRIPT= os.path.join(ROOT, "GPTzip", "gptzip.py")
BAK   = SCRIPT + ".bak"
MODEL = "C:/Users/28624/Informationtheory/Compression/gpt2_models"
os.chdir(ROOT)

# 修补 GPTzip（恢复+重打补丁，幂等）
if os.path.exists(BAK): shutil.copy(BAK, SCRIPT)
else: shutil.copy(SCRIPT, BAK)
src = open(SCRIPT, "r", encoding="utf-8").read()
src = src.replace('"gpt2"', f'"{MODEL}"')
src = src.replace('encoding="utf-8"', 'encoding="latin-1", newline=""')
open(SCRIPT, "w", encoding="utf-8").write(src)

SLICES = ["1KB", "10KB", "100KB"]   # 1MB 默认不跑，太慢；想跑就加进去
rows = []
for SLICE in SLICES:
    print(f"\n========== {SLICE} ==========")
    src_txt = f"data/enwik8_{SLICE}.txt"
    gpz_dbl = f"data/enwik8_{SLICE}.gpz.gpz"
    dec_txt = f"data/enwik8_{SLICE}.dec.txt"
    if not os.path.exists(src_txt) and os.path.exists(f"data/enwik8_{SLICE}.bin"):
        shutil.copy(f"data/enwik8_{SLICE}.bin", src_txt)
    for f in [f"data/enwik8_{SLICE}.gpz", gpz_dbl, dec_txt]:
        if os.path.exists(f): os.remove(f)

    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-z", src_txt, "-o", f"data/enwik8_{SLICE}.gpz"], check=True)
    t_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-u", gpz_dbl, "-o", dec_txt], check=True)
    t_d = time.perf_counter() - t0

    raw = open(src_txt, "rb").read()
    dec = open(dec_txt, "rb").read()
    ok  = hashlib.md5(raw).hexdigest() == hashlib.md5(dec).hexdigest()
    comp = os.path.getsize(gpz_dbl)
    bpb  = comp * 8 / len(raw)
    thr  = (len(raw) / 1e6) / max(t_c, 1e-6)
    row = ["GPTzip", SLICE, len(raw), comp, round(bpb,4),
           round(t_c,4), round(t_d,4), round(thr,4), "通过" if ok else "失败"]
    print(row)
    rows.append(row)

with open("results_llm.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerows(rows)
print("\n已写入 results_llm.csv")
