from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time
import re

# ==========================================
# 設定
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# バッチサイズの設定 (GPUのメモリに応じて調整)
BATCH_SIZE = 16
# 💡 T5-SmallやT5-BaseモデルベースのGECモデルなので、大きめのバッチサイズが使えます。

# モデルとトークナイザーの読み込み（変更なし）
MODEL_NAME = "prithivida/grammar_error_correcter_v1"
start_time_load = time.time()
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"Model loaded successfully in {time.time() - start_time_load:.2f} seconds.\n")

# ファイル名
INPUT_FILE = "essays_output_qwen25_7b.txt"
OUTPUT_FILE = "corrected_optimized.txt"

# ==========================================
# ステップ1: 入力ファイルの読み込みと文単位への分割
# ==========================================
# Qwenの出力フォーマットからエッセイの本文だけを抽出し、さらに文単位に分割する関数
def extract_and_split_sentences(filepath):
    """
    ファイルからエッセイを読み込み、不要なヘッダーを除去し、文単位でリスト化する。
    """
    sentences_list = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return []

    # 1. <<< Essay X >>> と --- で挟まれたエッセイ本文を抽出
    # re.DOTALL: . が改行文字も含むようにする
    essay_texts = re.findall(r"<<< Essay \d+ >>>\n(.*?)\n---", content, re.DOTALL)
    
    # 2. エッセイ本文を文単位に分割（簡易的な句読点分割）
    for text in essay_texts:
        # 句点(.)、疑問符(?)、感嘆符(!)で分割
        # re.splitはデリミタも含むようにカッコで囲んでいます
        text = text.replace('\n', ' ') # 改行をスペースに置換
        s_list = re.split(r'([.?!])\s*', text.strip())
        
        # 分割結果の結合とクリーンアップ
        sentence = ""
        for item in s_list:
            if item in ['.', '?', '!']:
                # 句読点を直前の文に結合し、文リストに追加
                sentences_list.append((sentence + item).strip())
                sentence = "" # sentenceをリセット
            else:
                sentence += item
        if sentence.strip():
             # 最後に残った文を追加
            sentences_list.append(sentence.strip())

    # 非常に短い文や空の文を除去
    return [s for s in sentences_list if len(s.split()) >= 3]

all_sentences = extract_and_split_sentences(INPUT_FILE)
total_sentences = len(all_sentences)

if total_sentences == 0:
    print("No valid sentences extracted. Exiting.")
    # モデルを削除してメモリを解放
    del model
    torch.cuda.empty_cache()
    # 必要に応じてファイルを生成して終了
    with open(OUTPUT_FILE, "w") as f:
        f.write("No sentences processed.")
    
# ==========================================
# ステップ2: バッチ処理と訂正の実行
# ==========================================
print(f"Total sentences extracted and ready for correction: {total_sentences}")
start_time_gen = time.time()
corrected_sentences = []

for i in range(0, total_sentences, BATCH_SIZE):
    # バッチを取得
    batch = all_sentences[i:i + BATCH_SIZE]
    
    # GECのプロンプトを付与
    input_texts = ["gec: " + s for s in batch]
    
    # バッチトークン化
    input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    # モデル生成（バッチ処理）
    outputs = model.generate(
        **input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    
    # デコード
    batch_corrected = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    corrected_sentences.extend(batch_corrected)
    
    # 進捗表示
    print(f"Processed {i + len(batch)} / {total_sentences} sentences...")

total_time = time.time() - start_time_gen
print(f"\nCorrection completed. Total sentences: {total_sentences}")
print(f"Total correction time: {total_time:.2f} seconds.")
print(f"Average time per sentence: {total_time / total_sentences:.4f} seconds.")

# ==========================================
# ステップ3: 訂正結果の出力
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for original, corrected in zip(all_sentences, corrected_sentences):
        f.write(f"Original: {original}\n")
        f.write(f"Corrected: {corrected}\n")
        f.write("-" * 50 + "\n")

print(f"\nすべての訂正結果を '{OUTPUT_FILE}' に保存しました。")

# モデルを削除してメモリを解放
del model
torch.cuda.empty_cache()