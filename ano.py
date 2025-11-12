from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os
from glob import glob
from tqdm import tqdm

# ==============================
# 設定 (Settings)
# ==============================
# 誤り文 (Source) が格納されている正解M2ファイルのパス
REF_M2_PATH = "conll14st-test-data/noalt/official-2014.0.m2" 

# 出力ファイル (Hypothesis: LLMが修正した文のみを保存)
OUTPUT_HYP_FILE = "llm_generated_hyp.txt"
CHECKPOINT_FILE = "checkpoint_conll14.json"

# ==============================
# モデル準備 (Model Setup)
# ==============================
# CUDAが利用可能かチェック
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2-7B-instruct"

# モデルとトークナイザーのロード（モデルは自動でGPUにマッピングされます）
tokenizer = AutoTokenizer.from_pretrained(model_id)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    # エラーが発生した場合、CPUでフォールバック (ただし、大規模モデルでは低速)
    print(f"CUDAエラーまたはメモリ不足: {e}. CPUにフォールバックします。")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu"
    )

# ==============================
# LLM用プロンプト (訂正文のみを要求)
# ==============================
# LLMに「元の文を訂正し、訂正された文のみを出力する」という役割を与えます。
PROMPT_TEMPLATE = """You are an expert grammatical error correction (GEC) system.
Your task is to correct the 'Original Sentence' below.
Return ONLY the corrected sentence. Do not add any explanation, quotes, or formatting.

Original Sentence:
{original}

Corrected Sentence:
"""

# ==============================
# M2ファイルからのオリジナルテキスト抽出
# (正解M2ファイルから元の文 (S行) のみを抽出)
# ==============================
def extract_texts_from_m2(m2_path):
    """
    M2ファイルから 'S' で始まる行 (元の文) のみを行ごとに抽出する。
    """
    texts = []
    try:
        with open(m2_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 'S 'で始まる行はオリジナルの誤り文
                if line.startswith('S '):
                    # 'S 'の後のテキストを抽出
                    texts.append(line[2:].strip())
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {m2_path}")
        return []
    except Exception as e:
        print(f"M2ファイル読み込み中に予期せぬエラーが発生しました: {e}")
        return []
        
    return texts

# ==============================
# LLMによる訂正文生成
# ==============================
def generate_correction(orig):
    """
    LLMに元の文を渡し、訂正された文のみを生成させる。
    """
    prompt = PROMPT_TEMPLATE.format(original=orig)

    # トークン化とモデル生成
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0, # 決定論的にするため
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 1. プロンプト部分の除去と整形
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()
    
    # 2. 複数の改行や引用符の除去（保険）
    cleaned_correction = re.sub(r'["\'\n]', '', output_text).strip()
    
    return cleaned_correction

# ==============================
# メイン処理 (Main Execution)
# ==============================
if __name__ == "__main__":
    print(f"デバイス: {device}")
    print(f"モデル: {model_id}")

    # 1. 元テキスト読み込み
    print(f"\n1. M2ファイルから誤り文を抽出中: {REF_M2_PATH}")
    original_texts = extract_texts_from_m2(REF_M2_PATH)
    
    if not original_texts:
        print("処理を中断します。M2ファイルのパスを確認するか、内容を確認してください。")
        exit()

    min_len = len(original_texts)
    print(f"抽出された文の数: {min_len} 件")

    # 2. チェックポイント読み込み
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        start_idx = len(checkpoint)
        print(f"\n2. 再開：{start_idx} 件目から訂正生成を開始")
    else:
        checkpoint = []
        start_idx = 0
        # 出力ファイルを新規作成する場合は、既存のファイルを削除
        if os.path.exists(OUTPUT_HYP_FILE):
            os.remove(OUTPUT_HYP_FILE)
            print(f"既存の出力ファイル {OUTPUT_HYP_FILE} を削除しました。")


    # 3. LLMによる訂正文生成
    for idx in tqdm(range(start_idx, min_len), desc="Correction Generation"):
        orig_text = original_texts[idx]
        
        try:
            # LLMに訂正文を生成させる
            corr_text = generate_correction(orig_text)

        except Exception as e:
            print(f"\n[{idx+1}/{min_len}] エラー発生: {e} -- スキップし、元の文を出力します。")
            # エラー時は元の文を訂正文として扱い、評価に含める (0スコアになるが構造は維持される)
            corr_text = orig_text 
        
        # チェックポイントに追加
        checkpoint.append({
            "index": idx,
            "original": orig_text,
            "corrected": corr_text
        })

        # チェックポイント随時保存
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        # 出力仮説ファイルに訂正文を随時保存 (1行1文)
        with open(OUTPUT_HYP_FILE, "a", encoding="utf-8") as f:
            f.write(corr_text + "\n")

    print(f"\n訂正文の生成完了：{OUTPUT_HYP_FILE}")
    print(f"チェックポイント保存済み：{CHECKPOINT_FILE}")

    # 4. 最終評価の案内
    print("\n=======================================================")
    print("訂正文の生成が完了しました。")
    print("次のステップは、ERRANTツールを用いた評価です。")
    print("この評価は3段階で実行する必要があります:")
    print("\n[ステップ 1] 原文ファイルの準備 (original.txtの作成)")
    print("  コマンド: grep \"^S \" conll14st-test-data/noalt/official-2014.0.m2 | sed 's/^S //g' > original.txt")
    print("\n[ステップ 2] LLMの出力からM2ファイルを作成 (errant_m2を使用)")
    print("  **コマンド: errant_m2 -orig original.txt -cor llm_generated_hyp.txt -out llm_annotated_hyp.m2**")
    print("\n[ステップ 3] M2ファイル同士を比較 (errant_compareを使用)")
    print("  コマンド: errant_compare -hyp llm_annotated_hyp.m2 -ref conll14st-test-data/noalt/official-2014.0.m2")
    print("=======================================================")