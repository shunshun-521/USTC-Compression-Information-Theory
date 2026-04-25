import os, shutil
src = "data/enwik8_1MB.bin"
dst = "data/enwik8_1MB.txt"
if not os.path.exists(dst) and os.path.exists(src):
    shutil.copy(src, dst)
    print(f"已复制 {src} -> {dst}")
elif os.path.exists(dst):
    print(f"{dst} 已存在")
else:
    # 直接从 enwik8 切
    with open("enwik8","rb") as f: d = f.read(1048576)
    with open(dst,"wb") as f: f.write(d)
    print(f"已生成 {dst}, {len(d)} B")
