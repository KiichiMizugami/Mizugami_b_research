import os
import re

# ディレクトリのパスを指定
data_dir = "ICNALE_EE_3.1/ICNALE_EE_3.1/EE_0_Unclassified_Unmerged"

# 出力ファイル
orig_out = "original.txt"
edit_out = "gold.txt"

# 出力ファイルを初期化
with open(orig_out, "w") as f1, open(edit_out, "w") as f2:
    pass

# ファイル名のペアを探す
files = os.listdir(data_dir)
orig_files = sorted([f for f in files if f.endswith("_ORIG.txt")])
edit_files = sorted([f for f in files if f.endswith("_EDIT.txt")])

# 名前のベース部分を抽出して対応付け
orig_bases = {re.sub(r"_ORIG\.txt$", "", f): f for f in orig_files}
edit_bases = {re.sub(r"_EDIT\.txt$", "", f): f for f in edit_files}

paired = sorted(set(orig_bases.keys()) & set(edit_bases.keys()))
print(f"Found {len(paired)} matching essay pairs.")

# ペアごとに読み込んで書き込み
count = 0
with open(orig_out, "a") as f1, open(edit_out, "a") as f2:
    for base in paired:
        orig_path = os.path.join(data_dir, orig_bases[base])
        edit_path = os.path.join(data_dir, edit_bases[base])
        with open(orig_path, "r", encoding="utf-8", errors="ignore") as o, \
             open(edit_path, "r", encoding="utf-8", errors="ignore") as e:
            orig_text = o.read().strip()
            edit_text = e.read().strip()
            if orig_text and edit_text:
                f1.write(orig_text + "\n")
                f2.write(edit_text + "\n")
                count += 1

print(f" Finished writing {count} essay pairs to original.txt and gold.txt")
