# ==========================================
# ライブラリのインポート
# ==========================================
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 設定
# ==========================================
# キャッシュ先を自分のホームに変更（権限エラー対策）
os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/huggingface")

# 出力ファイル
output_file = "essays_Qwen3_CHN.txt"

# GPU設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# モデル指定
# ==========================================
model_id = "Qwen/Qwen3-14B"

# トークナイザとモデルの読み込み
print(" Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print(" Model loaded successfully.\n")

# ==========================================
# プロンプト（日本人英語学習者の作文生成）
# ==========================================
prompt = """You are a English learner.
 Your native language is Chinese.
 Please write an English essay .
 The essay should answer the following topic:
 "It is important for college students to have a part-time job."
 
 Write in simple English.
 please generate one essay of about 200-300 words.
 """

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

# ==========================================
# 生成パラメータ
# ==========================================
num_essays = 20  # 生成するエッセイ数

outputs = model.generate(
    **inputs,
    max_new_tokens=350,
    do_sample=True,
    temperature=0.7,  # 多様性を少し上げる
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=num_essays
)

# ==========================================
# 文ごとに改行する関数
# ==========================================
def split_into_sentences(text):
    # ピリオド・感嘆符・疑問符で区切り、後に空白があれば改行
    sentences = re.split(r'([.!?])\s+', text)
    # split で分割すると区切り文字が別要素になるので結合
    result = []
    for i in range(0, len(sentences)-1, 2):
        result.append(sentences[i] + sentences[i+1])
    # 末尾の文章が残る場合
    if len(sentences) % 2 != 0:
        result.append(sentences[-1])
    return result

# ==========================================
# 出力処理
# ==========================================
print("--- LLMによる日本語母語話者の英作文シミュレーション（全20個） ---")
print("-" * 60)

# ファイルを最初に空にする
with open(output_file, "w", encoding="utf-8") as f:
    f.write("")

for i, output in enumerate(outputs, 1):
    generated_tokens = output[input_length:]
    essay = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 文ごとに分割
    sentences = split_into_sentences(essay)
    
    # 画面出力
    print(f"\n<<< Essay {i} >>>")
    for s in sentences:
        print(s)
    print("-" * 60)

    # ファイル出力
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"<<< Essay {i} >>>\n")
        for s in sentences:
            f.write(s + "\n")
        f.write("-" * 60 + "\n")

print(f"\n生成完了。すべてのエッセイを '{output_file}' に保存しました。")
