import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        # Q, K, Vの変換行列
        self.W_q = nn.Linear(E, H * D, bias=False)
        self.W_k = nn.Linear(E, H * D, bias=False)
        self.W_v = nn.Linear(E, H * D, bias=False)
        self.W_o = nn.Linear(H * D, E, bias=False)  # 出力変換行列

        # Dropout
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # x: (B, C, E)
        B, C, E = x.shape  # バッチサイズ、コンテキスト長、埋め込み次元
        H, D = self.n_head, self.head_dim  # ヘッド数、ヘッドの次元

        # Q, K, V の計算
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 各ヘッドに分割して並び替え
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        # 計算
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (D**0.5)

        # マスク処理
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention重み
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V)

        # ヘッドの結合
        hidden = hidden.transpose(1, 2)
        hidden = hidden.contiguous().view(B, C, H * D)

        # 出力変換
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FFN(nn.Module):
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)  # 中間層の次元

        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        # 残差接続
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_context_len,
        embed_dim,
        n_head,
        n_layer,
        ff_dim,
        dropout_rate,
    ):
        super().__init__()
        self.vocab_size = vocab_size  # 語彙サイズ
        self.max_context_len = max_context_len  # 最大コンテキスト長
        self.embed_dim = embed_dim  # 埋め込み次元数
        self.n_head = n_head  # Attentionのヘッド数
        self.n_layer = n_layer  # Transformerブロック数
        self.ff_dim = ff_dim  # FFNの隠れ層サイズ
        self.dropout_rate = dropout_rate  # ドロップアウト率

        # 埋め込み層
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Transfomerブロック
        self.blocks = nn.ModuleList(
            [Block(embed_dim, n_head, ff_dim, dropout_rate) for _ in range(n_layer)]
        )

        # 出力層
        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        # 重み共有
        self.embed.weight = self.unembed.weight

        # 重みの初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, C = ids.shape  # B: バッチサイズ, C: コンテキスト長
        device = ids.device

        # 埋め込み
        pos = torch.arange(0, C, dtype=torch.long, device=device)
        emb = self.embed(ids)
        pos_emb = self.pos_embed(pos)
        x = self.dropout(emb + pos_emb)

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 出力
        logits = self.unembed(x)  # (B, C, vocab_size)

        return logits

    def save(self, file_path):
        checkpoint = {
            "model_state_dict": self.state_dict(),  # モデルの重み
            "vocab_size": self.vocab_size,
            "max_context_len": self.max_context_len,
            "embed_dim": self.embed_dim,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path, device="cpu"):
        checkpoint = torch.load(file_path, map_location=device)

        model = cls(
            vocab_size=checkpoint["vocab_size"],
            max_context_len=checkpoint["max_context_len"],
            embed_dim=checkpoint["embed_dim"],
            n_head=checkpoint["n_head"],
            n_layer=checkpoint["n_layer"],
            ff_dim=checkpoint["ff_dim"],
            dropout_rate=checkpoint["dropout_rate"],
        )
        # 重みの読み込み
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model
