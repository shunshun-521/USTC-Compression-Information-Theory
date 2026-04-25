# ============ GPTzip 一键流水线 ============
import os, sys, hashlib, subprocess, glob, shutil, time

ROOT  = r"C:\Users\28624\Informationtheory\Compression"
PY    = sys.executable
SCRIPT= os.path.join(ROOT, "GPTzip", "gptzip.py")
BAK   = SCRIPT + ".bak"
MODEL = "C:/Users/28624/Informationtheory/Compression/gpt2_models"
os.chdir(ROOT)

# ---- 1. 修补 GPTzip ----
print("[1/4] 修补 GPTzip...")
if os.path.exists(BAK):
    shutil.copy(BAK, SCRIPT)                              # 先还原到原始版
else:
    shutil.copy(SCRIPT, BAK)                              # 第一次执行：备份原始版
src = open(SCRIPT, "r", encoding="utf-8").read()
src = src.replace('"gpt2"', f'"{MODEL}"')                 # 模型路径
src = src.replace('encoding="utf-8"', 'encoding="latin-1", newline=""')
open(SCRIPT, "w", encoding="utf-8").write(src)
print("    完成。")

# ---- 2. 选择数据切片（先做 1KB） ----
SLICE = "1KB"
src_txt  = f"data/enwik8_{SLICE}.txt"
gpz_dbl  = f"data/enwik8_{SLICE}.gpz.gpz"   # GPTzip 强制双扩展名
dec_txt  = f"data/enwik8_{SLICE}.dec.txt"

# 如果 .txt 切片还不存在，从 .bin 复制一份
if not os.path.exists(src_txt) and os.path.exists(f"data/enwik8_{SLICE}.bin"):
    shutil.copy(f"data/enwik8_{SLICE}.bin", src_txt)

# ---- 3. 清理旧产物 + 压缩 + 解压 ----
for f in [f"data/enwik8_{SLICE}.gpz", gpz_dbl, dec_txt]:
    if os.path.exists(f): os.remove(f)

print(f"[2/4] 压缩 {src_txt} ...")
t0 = time.perf_counter()
subprocess.run([PY, SCRIPT, "-z", src_txt, "-o", f"data/enwik8_{SLICE}.gpz"], check=True)
t_c = time.perf_counter() - t0

print(f"[3/4] 解压 {gpz_dbl} ...")
t0 = time.perf_counter()
subprocess.run([PY, SCRIPT, "-u", gpz_dbl, "-o", dec_txt], check=True)
t_d = time.perf_counter() - t0

# ---- 4. 校验 + BPB ----
print("[4/4] 校验结果...")
raw = open(src_txt, "rb").read()
dec = open(dec_txt, "rb").read()
ok  = hashlib.md5(raw).hexdigest() == hashlib.md5(dec).hexdigest()
comp_size = os.path.getsize(gpz_dbl)
print()
print("=" * 50)
print(f"切片:       {SLICE}")
print(f"原始大小:   {len(raw)} B")
print(f"解压大小:   {len(dec)} B")
print(f"压缩大小:   {comp_size} B")
print(f"BPB:        {comp_size * 8 / len(raw):.4f}")
print(f"压缩耗时:   {t_c:.2f} s")
print(f"解压耗时:   {t_d:.2f} s")
print(f"无损校验:   {'通过 ✓' if ok else '失败 ✗'}")
print("=" * 50)
