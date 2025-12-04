import os
import re

INPUT_DIR = "ICNALE_WE_2.6/WE_2_Classified_Merged"
OUTPUT_DIR = "ICNALE_WE_2.6_by_L1_File"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pattern = re.compile(r"WE_([A-Z]{3})_")  

l1_contents = {}

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".txt"):
        continue

    match = pattern.search(fname)
    if not match:
        print("L1抽出できないファイル:", fname)
        continue

    l1 = match.group(1)

    with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
        text = f.read()

    if l1 not in l1_contents:
        l1_contents[l1] = []

    # ファイル名を区切りとして追加
    l1_contents[l1].append(f"=== {fname} ===\n{text}\n")
    print(f"{fname} を {l1} に追加")

for l1, texts in l1_contents.items():
    out_path = os.path.join(OUTPUT_DIR, f"{l1}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"{l1}.txt を作成")

print("完了！")
