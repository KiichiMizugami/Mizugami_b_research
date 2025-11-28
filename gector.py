from gector.gec_model import GecBERTModel

# 英語モデルのパス
model = GecBERTModel(
    vocab_path=None,  # 英語モデルでは vocab は不要
    model_paths=['~/.gector_models/en_gector/bert_0_gectorv2.th'],
    max_len=128
)

sentences = ["This are a test sentence.", "He go to school yesterday."]
preds = model.handle_batch(sentences)

for s, p in zip(sentences, preds):
    print("入力:", s)
    print("訂正:", p)
