import errant
import sys

# ==========================================
# Errantオブジェクトの初期化（v3以降対応）
# ==========================================
try:
    annotator = errant.load("en")  # ✅ spaCy + merger + aligner を構築
    nlp = annotator.nlp
except OSError:
    print("Error: The spaCy model 'en_core_web_sm' was not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred during Errant initialization: {e}")
    sys.exit()

# ==========================================
# 入力ファイル設定
# ==========================================
SOURCE_FILE = "original.txt"             # 誤りを含む元文
TARGET_FILE = "gold.txt"                 # 正解文（人手訂正）
PREDICTED_FILE = "corrected_optimized.txt"  # モデルの訂正文

print("Errant: Starting evaluation...")

# ==========================================
# ファイルの読み込み
# ==========================================
try:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        sources = f.read().splitlines()
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        targets = f.read().splitlines()
    with open(PREDICTED_FILE, "r", encoding="utf-8") as f:
        predictions = f.read().splitlines()
except FileNotFoundError as e:
    print(f"Error: Input file not found: {e.filename}")
    sys.exit()

# 文数チェック
if not (len(sources) == len(targets) == len(predictions)):
    print(f"Error: Sentence count mismatch — orig={len(sources)}, gold={len(targets)}, pred={len(predictions)}")
    sys.exit()

print(f"Processing {len(sources)} sentence pairs...")

# ==========================================
# gold.m2 と hyp.m2 を生成
# ==========================================
with open("gold.m2", "w", encoding="utf-8") as gold_m2, \
     open("hyp.m2", "w", encoding="utf-8") as hyp_m2:

    for source, target, pred in zip(sources, targets, predictions):
        orig = nlp(source)
        cor_gold = nlp(target)
        cor_pred = nlp(pred)

        # gold
        gold_edits = annotator.annotate(orig, cor_gold)
        gold_m2.write("S " + source + "\n")
        for edit in gold_edits:
            gold_m2.write(edit.to_m2() + "\n")
        gold_m2.write("\n")

        # hyp
        hyp_edits = annotator.annotate(orig, cor_pred)
        hyp_m2.write("S " + source + "\n")
        for edit in hyp_edits:
            hyp_m2.write(edit.to_m2() + "\n")
        hyp_m2.write("\n")

print("✅ gold.m2 と hyp.m2 を生成しました。")
print("次に以下を実行してください：")
print("----------------------------------------------------")
print("errant_compare -hyp hyp.m2 -ref gold.m2")
print("----------------------------------------------------")
