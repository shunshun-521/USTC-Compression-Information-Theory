import os, sys, hashlib, subprocess, shutil, time, csv

ROOT  = r"C:\Users\28624\Informationtheory\Compression"
PY    = sys.executable
SCRIPT= os.path.join(ROOT, "GPTzip", "gptzip.py")
os.chdir(ROOT)

SLICE = "1MB"
src_txt = f"data/enwik8_{SLICE}.txt"
gpz_dbl = f"data/enwik8_{SLICE}.gpz.gpz"
dec_txt = f"data/enwik8_{SLICE}.dec.txt"
for f in [f"data/enwik8_{SLICE}.gpz", gpz_dbl, dec_txt]:
    if os.path.exists(f): os.remove(f)

print(f"[start] {time.strftime('%H:%M:%S')}  压缩 {SLICE}...")
t0 = time.perf_counter()
subprocess.run([PY, SCRIPT, "-z", src_txt, "-o", f"data/enwik8_{SLICE}.gpz"], check=True)
t_c = time.perf_counter() - t0
print(f"[mid]   {time.strftime('%H:%M:%S')}  压缩完成，耗时 {t_c/60:.1f} 分钟，开始解压...")

t0 = time.perf_counter()
subprocess.run([PY, SCRIPT, "-u", gpz_dbl, "-o", dec_txt], check=True)
t_d = time.perf_counter() - t0

raw = open(src_txt, "rb").read()
dec = open(dec_txt, "rb").read()
ok  = hashlib.md5(raw).hexdigest() == hashlib.md5(dec).hexdigest()
comp = os.path.getsize(gpz_dbl)

row = ["GPTzip", SLICE, len(raw), comp, round(comp*8/len(raw),4),
       round(t_c,4), round(t_d,4), round((len(raw)/1e6)/max(t_c,1e-6),6),
       "通过" if ok else "失败"]
print(f"[done]  {time.strftime('%H:%M:%S')}  {row}")

# 追加到现有 LLM 结果
with open("results_llm.csv", "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
print("已追加到 results_llm.csv")
