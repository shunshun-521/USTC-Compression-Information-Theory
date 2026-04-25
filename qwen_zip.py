"""
Qwen-based 无损文本压缩器
用法:
    python qwen_zip.py -z input.txt output.qwz   # 压缩
    python qwen_zip.py -u output.qwz output.txt  # 解压
"""
import sys, os, struct, time, argparse
import torch
import gptzip
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"./qwen25_05b_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading Qwen2.5-0.5B from {MODEL_DIR} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Qwen 没有专门的 bos_token，借用 eos 当 BOS（gptzip 库要求必须有）
    if tok.bos_token_id is None:
        tok.bos_token = tok.eos_token   # 通常是 <|endoftext|> (id=151643)
        print(f"  借用 eos_token 作为 bos_token: id={tok.bos_token_id}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    ).to(DEVICE).eval()
    return tok, model

def compress(in_path, out_path):
    tok, model = load_model()
    raw = open(in_path, "rb").read()
    raw_text = raw.decode("latin-1")
    print(f"Encoding {len(raw)} bytes...")
    t0 = time.perf_counter()
    coder = gptzip.ArithmeticCoder(lm=model, tokenizer=tok)
    code, num_padded_bits = coder.encode(raw_text, return_num_padded_bits=True)
    t = time.perf_counter() - t0
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", len(raw), num_padded_bits))
        f.write(code)
    print(f"OK: 原始 {len(raw)} B -> 压缩 {os.path.getsize(out_path)} B  耗时 {t:.2f}s")
    print(f"BPB: {os.path.getsize(out_path)*8/len(raw):.4f}")

def decompress(in_path, out_path):
    tok, model = load_model()
    with open(in_path, "rb") as f:
        raw_len, num_padded_bits = struct.unpack("<II", f.read(8))
        code = f.read()
    print(f"Decoding to {raw_len} bytes...")
    t0 = time.perf_counter()
    coder = gptzip.ArithmeticCoder(lm=model, tokenizer=tok)
    decoded = coder.decode(code, num_padded_bits=num_padded_bits)
    t = time.perf_counter() - t0
    out_bytes = decoded.encode("latin-1")[:raw_len]
    with open(out_path, "wb") as f:
        f.write(out_bytes)
    print(f"OK: 解压 {len(out_bytes)} B  耗时 {t:.2f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-z", nargs=2, metavar=("INPUT","OUTPUT"))
    g.add_argument("-u", nargs=2, metavar=("INPUT","OUTPUT"))
    args = ap.parse_args()
    if args.z: compress(*args.z)
    else:      decompress(*args.u)
