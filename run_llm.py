"""
通用 LLM 压缩批量脚本
用法: python run_llm.py <model_path> <model_label> <slice1> <slice2> ...
例:  python run_llm.py ./gpt2_medium_models GPTzip-medium 1KB 10KB 100KB
"""
import os, sys, hashlib, subprocess, shutil, time, csv, re

ROOT  = r"C:\Users\28624\Informationtheory\Compression"
PY    = sys.executable
SCRIPT= os.path.join(ROOT, "GPTzip", "gptzip.py")
BAK   = SCRIPT + ".bak"
os.chdir(ROOT)

if len(sys.argv) < 4:
    print("用法: python run_llm.py <model_path> <model_label> <slice1> [slice2 ...]")
    sys.exit(1)

MODEL_PATH  = os.path.abspath(sys.argv[1]).replace("\\", "/")
MODEL_LABEL = sys.argv[2]
SLICES      = sys.argv[3:]

# ---- 修补 GPTzip：把模型路径替换为指定的 ----
print(f"[patch] 设置模型路径 = {MODEL_PATH}")
if not os.path.exists(BAK): shutil.copy(SCRIPT, BAK)
src = open(BAK, "r", encoding="utf-8").read()       # 总是从备份恢复，避免叠加污染
src = re.sub(r'"gpt2"', f'"{MODEL_PATH}"', src)
src = src.replace('encoding="utf-8"', 'encoding="latin-1", newline=""')
open(SCRIPT, "w", encoding="utf-8").write(src)

# ---- 跑批量 ----
out_csv = f"results_{MODEL_LABEL}.csv"
rows = []
for SLICE in SLICES:
    print(f"\n========== {MODEL_LABEL} on {SLICE} ==========")
    src_txt = f"data/enwik8_{SLICE}.txt"
    gpz_dbl = f"data/{MODEL_LABEL}_{SLICE}.gpz.gpz"
    gpz_sgl = f"data/{MODEL_LABEL}_{SLICE}.gpz"
    dec_txt = f"data/{MODEL_LABEL}_{SLICE}.dec.txt"
    if not os.path.exists(src_txt):
        bin_path = f"data/enwik8_{SLICE}.bin"
        if os.path.exists(bin_path): shutil.copy(bin_path, src_txt)
    for f in [gpz_sgl, gpz_dbl, dec_txt]:
        if os.path.exists(f): os.remove(f)

    print(f"[start] {time.strftime('%H:%M:%S')} 压缩 {SLICE}...")
    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-z", src_txt, "-o", gpz_sgl], check=True)
    t_c = time.perf_counter() - t0

    print(f"[mid]   {time.strftime('%H:%M:%S')} 压缩完成 {t_c/60:.2f} min, 解压中...")
    t0 = time.perf_counter()
    subprocess.run([PY, SCRIPT, "-u", gpz_dbl, "-o", dec_txt], check=True)
    t_d = time.perf_counter() - t0

    raw = open(src_txt, "rb").read()
    dec = open(dec_txt, "rb").read()
    ok  = hashlib.md5(raw).hexdigest() == hashlib.md5(dec).hexdigest()
    comp = os.path.getsize(gpz_dbl)
    bpb  = comp * 8 / len(raw)
    thr  = (len(raw) / 1e6) / max(t_c, 1e-6)

    row = [MODEL_LABEL, SLICE, len(raw), comp, round(bpb,4),
           round(t_c,4), round(t_d,4), round(thr,6), "通过" if ok else "失败"]
    print(f"[done]  {time.strftime('%H:%M:%S')} {row}")
    rows.append(row)

with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["method","slice","raw_B","comp_B","BPB","t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerows(rows)
print(f"\n已写入 {out_csv}")
