import os, random, inspect
os.makedirs("data_exp3", exist_ok=True)

# 1. 英文维基（前 1KB）
data = open("enwik8","rb").read(1024)
open("data_exp3/wiki_en.txt","wb").write(data)
print(f"[1/4] wiki_en.txt: {os.path.getsize('data_exp3/wiki_en.txt')} B")

# 2. Python 代码（约 1KB CPython 标准库源码）
import random as r, json, csv, hashlib as h
src_blocks = []
for mod in [r, json, csv, h]:
    try: src_blocks.append(inspect.getsource(mod))
    except: pass
src = "\n".join(src_blocks).encode("utf-8")[:1024]
while True:
    try: src.decode("utf-8"); break
    except UnicodeDecodeError: src = src[:-1]
open("data_exp3/code_py.txt","wb").write(src)
print(f"[2/4] code_py.txt: {os.path.getsize('data_exp3/code_py.txt')} B")

# 3. 中文文本（信息论领域中文段落，重复填充到 1KB）
zh_seed = "信息论是应用数学的一个分支主要研究信息的量化存储和通信这个领域由克劳德香农在二十世纪四十年代奠定基础香农在一九四八年发表的论文通信的数学理论中提出了信息熵的概念从此奠定了现代信息论的基础信息熵衡量随机变量的不确定性单位通常是比特"
zh = (zh_seed * 20).encode("utf-8")[:1024]
while True:
    try: zh.decode("utf-8"); break
    except UnicodeDecodeError: zh = zh[:-1]
open("data_exp3/text_zh.txt","wb").write(zh)
print(f"[3/4] text_zh.txt: {os.path.getsize('data_exp3/text_zh.txt')} B")
print(f"  实际字符数: {len(zh.decode('utf-8'))} 个汉字")

# 4. 伪随机字节（理论 BPB ≈ 8）
random.seed(42)
rb = bytes(random.randint(0,255) for _ in range(1024))
open("data_exp3/random.bin","wb").write(rb)
print(f"[4/4] random.bin: {os.path.getsize('data_exp3/random.bin')} B")

print("\n所有 4 类数据已就绪 (data_exp3/)")
