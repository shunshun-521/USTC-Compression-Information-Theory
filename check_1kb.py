import os, hashlib, glob
raw = open("data/enwik8_1KB.txt","rb").read()
candidates = glob.glob("data/enwik8_1KB.dec*")
print("候选解压文件:", candidates)
for c in candidates:
    if os.path.isfile(c):
        dec = open(c,"rb").read()
        ok = hashlib.md5(raw).hexdigest() == hashlib.md5(dec).hexdigest()
        print(f"  {c}: 大小={len(dec)} B, 无损={'通过' if ok else '失败'}")
comp = "data/enwik8_1KB.gpz.gpz"
print(f"原始: {len(raw)} B")
print(f"压缩: {os.path.getsize(comp)} B")
print(f"BPB:  {os.path.getsize(comp)*8/len(raw):.4f}")
