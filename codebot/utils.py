import torch
import torch.nn.functional as F


def get_device():
    """利用可能なデバイスを自動検出して返す"""
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 評価

    # 1.プロンプトをトークン化
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = ids.clone()  # 生成されたトークンを保持する変数

    # 2.トークン生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト長を超えた場合、古いトークンを切り捨てる
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len :]

        # 最後の位置のロジットを取得（次トークンの予測）
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 終了判定
        if next_id.item() == tokenizer.end_token_id:
            break

        # トークンの追加
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # デコード
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text
