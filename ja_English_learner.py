from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ==========================================
#  設定
# ==========================================
# キャッシュ先を自分のホームに変更（権限エラー対策）
os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/huggingface")

# 出力ファイル
output_file = "essays_output_qwen25_7b.txt"

# GPU設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
#  モデル指定
# ==========================================
# Qwen2.5-7B-Instruct を使用
model_id = "Qwen/Qwen2.5-7B-Instruct"

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
#  プロンプト（日本人英語学習者の作文生成）
# ==========================================
prompt = """You are a Japanese English learner.
Your native language is Japanese.
Please write an English essay as if you are a Japanese student learning English.
The essay should answer the following topic:

"It is important for college students to have a part-time job."

Write in simple English.
please generate one essay of about 200-300 words.

"""

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

# ==========================================
#  生成パラメータ
# ==========================================
num_essays = 20                                        

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
#  出力処理
# ==========================================
print("--- LLMによる日本語母語話者の英作文シミュレーション（全20個） ---")
print("-" * 60)

with open(output_file, "w", encoding="utf-8") as f:
    for i, output in enumerate(outputs, 1):
        generated_tokens = output[input_length:]
        essay = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 画面出力
        print(f"\n<<< Essay {i} >>>")
        print(essay)
        print("-" * 60)

        # ファイル出力
        f.write(f"<<< Essay {i} >>>\n")
        f.write(essay + "\n")
        f.write("-" * 60 + "\n")

print(f"\n 生成完了。すべてのエッセイを '{output_file}' に保存しました。")