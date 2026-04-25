"""跨领域文本压缩对比脚本（实验三）"""
import os, sys, hashlib, subprocess, time, csv, gzip, bz2, lzma, struct
import zstandard as zstd
import torch
import gptzip as _gptzip_lib
from transformers import AutoTokenizer, AutoModelForCausalLM

PY = sys.executable
ROOT = r"C:\Users\28624\Informationtheory\Compression"
GPTZIP_SCRIPT = os.path.join(ROOT, "GPTzip", "gptzip.py")
GPTZIP_BAK    = GPTZIP_SCRIPT + ".bak"
SMALL_PATH    = "C:/Users/28624/Informationtheory/Compression/gpt2_models"
MEDIUM_PATH   = "C:/Users/28624/Informationtheory/Compression/gpt2_medium_models"
QWEN_PATH     = "./qwen25_05b_models"
os.chdir(ROOT)

CORPORA = [
    ("wiki_en", "data_exp3/wiki_en.txt"),
    ("code_py", "data_exp3/code_py.txt"),
    ("text_zh", "data_exp3/text_zh.txt"),
    ("random",  "data_exp3/random.bin"),
]

# ---- 传统方法（Python 内置库）----
def run_traditional(name, raw):
    if name == "gzip":
        t0=time.perf_counter(); c=gzip.compress(raw, 9);   tc=time.perf_counter()-t0
        t0=time.perf_counter(); d=gzip.decompress(c);      td=time.perf_counter()-t0
    elif name == "bzip2":
        t0=time.perf_counter(); c=bz2.compress(raw, 9);    tc=time.perf_counter()-t0
        t0=time.perf_counter(); d=bz2.decompress(c);       td=time.perf_counter()-t0
    elif name == "xz":
        t0=time.perf_counter(); c=lzma.compress(raw, preset=9); tc=time.perf_counter()-t0
        t0=time.perf_counter(); d=lzma.decompress(c);      td=time.perf_counter()-t0
    elif name == "zstd":
        cctx=zstd.ZstdCompressor(level=22); dctx=zstd.ZstdDecompressor()
        t0=time.perf_counter(); c=cctx.compress(raw);      tc=time.perf_counter()-t0
        t0=time.perf_counter(); d=dctx.decompress(c);      td=time.perf_counter()-t0
    return c, d, tc, td

# ---- GPTzip 修补 ----
def patch_gptzip(model_path):
    import shutil
    if not os.path.exists(GPTZIP_BAK): shutil.copy(GPTZIP_SCRIPT, GPTZIP_BAK)
    src = open(GPTZIP_BAK, "r", encoding="utf-8").read()
    src = src.replace('"gpt2"', f'"{model_path}"')
    src = src.replace('encoding="utf-8"', 'encoding="latin-1", newline=""')
    open(GPTZIP_SCRIPT, "w", encoding="utf-8").write(src)

def run_gptzip(model_path, label, src_txt):
    patch_gptzip(model_path)
    base = f"data_exp3/{label}_{os.path.basename(src_txt).replace('.','_')}"
    gpz_dbl = base + ".gpz.gpz"
    gpz_sgl = base + ".gpz"
    dec = base + ".dec.txt"
    for f in [gpz_sgl, gpz_dbl, dec]:
        if os.path.exists(f): os.remove(f)
    t0 = time.perf_counter()
    subprocess.run([PY, GPTZIP_SCRIPT, "-z", src_txt, "-o", gpz_sgl], check=True,
                   capture_output=True)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    subprocess.run([PY, GPTZIP_SCRIPT, "-u", gpz_dbl, "-o", dec], check=True,
                   capture_output=True)
    td = time.perf_counter() - t0
    raw = open(src_txt, "rb").read()
    decb = open(dec, "rb").read()
    comp = open(gpz_dbl, "rb").read()
    return comp, decb, tc, td

# ---- Qwen ----
_qwen_cache = {}
def get_qwen():
    if "loaded" not in _qwen_cache:
        print("  [一次性加载 Qwen...]")
        tok = AutoTokenizer.from_pretrained(QWEN_PATH)
        if tok.bos_token_id is None:
            tok.bos_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, torch_dtype=torch.float16
        ).to("cuda").eval()
        _qwen_cache["tok"] = tok
        _qwen_cache["model"] = model
        _qwen_cache["loaded"] = True
    return _qwen_cache["tok"], _qwen_cache["model"]

def run_qwen(src_txt):
    tok, model = get_qwen()
    raw = open(src_txt, "rb").read()
    raw_text = raw.decode("latin-1")
    coder = _gptzip_lib.ArithmeticCoder(lm=model, tokenizer=tok)
    t0 = time.perf_counter()
    code, n_pad = coder.encode(raw_text, return_num_padded_bits=True)
    tc = time.perf_counter() - t0
    # 压缩 = 8 字节头 + code
    comp = struct.pack("<II", len(raw), n_pad) + code
    t0 = time.perf_counter()
    decoded = coder.decode(code, num_padded_bits=n_pad)
    td = time.perf_counter() - t0
    decb = decoded.encode("latin-1")[:len(raw)]
    return comp, decb, tc, td

# ---- 主循环 ----
rows = []
for corpus, path in CORPORA:
    print(f"\n========== {corpus} ({path}) ==========")
    raw = open(path, "rb").read()

    # 4 个传统方法
    for m in ["gzip","bzip2","xz","zstd"]:
        c, d, tc, td = run_traditional(m, raw)
        ok = hashlib.md5(raw).hexdigest()==hashlib.md5(d).hexdigest()
        bpb = len(c)*8/len(raw)
        thr = (len(raw)/1e6)/max(tc,1e-6)
        row = [corpus, m, len(raw), len(c), round(bpb,4),
               round(tc,4), round(td,4), round(thr,4),
               "通过" if ok else "失败"]
        print(f"  {m:8s}  BPB={bpb:6.4f}  comp={len(c):5d}B")
        rows.append(row)

    # GPT-2 small
    print("  GPT-2 small...")
    c, d, tc, td = run_gptzip(SMALL_PATH, "small", path)
    ok = hashlib.md5(raw).hexdigest()==hashlib.md5(d).hexdigest()
    bpb = len(c)*8/len(raw)
    thr = (len(raw)/1e6)/max(tc,1e-6)
    row = [corpus, "GPT-2-small", len(raw), len(c), round(bpb,4),
           round(tc,4), round(td,4), round(thr,6),
           "通过" if ok else "失败"]
    print(f"  GPT-2-small  BPB={bpb:6.4f}  comp={len(c):5d}B  耗时={tc:.1f}s")
    rows.append(row)

    # GPT-2 medium
    print("  GPT-2 medium...")
    c, d, tc, td = run_gptzip(MEDIUM_PATH, "medium", path)
    ok = hashlib.md5(raw).hexdigest()==hashlib.md5(d).hexdigest()
    bpb = len(c)*8/len(raw)
    thr = (len(raw)/1e6)/max(tc,1e-6)
    row = [corpus, "GPT-2-medium", len(raw), len(c), round(bpb,4),
           round(tc,4), round(td,4), round(thr,6),
           "通过" if ok else "失败"]
    print(f"  GPT-2-medium BPB={bpb:6.4f}  comp={len(c):5d}B  耗时={tc:.1f}s")
    rows.append(row)

    # Qwen
    print("  Qwen2.5-0.5B...")
    c, d, tc, td = run_qwen(path)
    ok = hashlib.md5(raw).hexdigest()==hashlib.md5(d).hexdigest()
    bpb = len(c)*8/len(raw)
    thr = (len(raw)/1e6)/max(tc,1e-6)
    row = [corpus, "Qwen2.5-0.5B", len(raw), len(c), round(bpb,4),
           round(tc,4), round(td,4), round(thr,6),
           "通过" if ok else "失败"]
    print(f"  Qwen2.5-0.5B BPB={bpb:6.4f}  comp={len(c):5d}B  耗时={tc:.1f}s")
    rows.append(row)

# ---- 写 CSV ----
with open("results_exp3.csv","w",newline="",encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["corpus","method","raw_B","comp_B","BPB",
                "t_compress_s","t_decompress_s","throughput_MBps","lossless"])
    w.writerows(rows)

# ---- 汇总打印 ----
print("\n" + "="*70)
print(f"{'corpus':10s} {'method':14s} {'BPB':>7s}  {'lossless':>8s}")
print("-"*70)
for r in rows:
    print(f"{r[0]:10s} {r[1]:14s} {r[4]:7.4f}  {r[8]:>8s}")
print("\n已写入 results_exp3.csv")
