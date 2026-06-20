from collections import defaultdict
import regex as re
from tqdm import tqdm
import pickle

def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)
    
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2  # Skip the next id since it's part of the merged pair
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def pretokenize(text):
    # GPT-2で使用されている正規表現パターン
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # step1: 特殊トークンで分割
    texts = input_text.split(end_token)

    #step2: 各テキスト片を事前トークン化
    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode('utf-8')))  # to ids

    # 基本語彙 + 終了トークン用1個を除いた分がマージ回数
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        # ペアの出現頻度を集計
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)
        
        if not counts:
            break  # No more pairs to merge

        # 最も頻度の高いペアを取得
        best_pair = max(counts, key=counts.get)  # Get the most frequent pair
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # マージ実行
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules

class BPETokenizer:
    @staticmethod
    def load_from(filepath):
        with open(filepath, 'rb') as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)
    
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)  # ID for the end token

        # idからバイト列への対応表 0-255
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}  # Initialize with single byte values
        # マージされたトークンは元のトークンのバイト列を連結
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode('utf-8')  # Add end token to the mapping

        self.vocab_size = len(self.id_to_bytes)
    
    def _encode_text(self, text):
        ids = list(text.encode('utf-8'))  # Start with byte values
        # 学習時の順序でマージルールを適用
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)        
        return ids
    
    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        texts = tqdm(texts, desc="Encoding") if show_progress else texts  # Show progress bar if specified

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    all_ids.extend(self._encode_text(pretoken))

        return all_ids
     
    def decode(self, ids):
        # 各トークンIDを対応するバイト列に変換
        byte_list = [self.id_to_bytes[id] for id in ids]
        text_bytes = b''.join(byte_list)
        text = text_bytes.decode('utf-8', errors='replace')  # Decode with error handling
        return text

# 事前トークン化対応のBPE学習
sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"

merge_rules = train_bpe(sample_text, vocab_size=270)
tokenizer = BPETokenizer(merge_rules)

# エンコード/デコード
text = "Say hello!"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)

# 各トークンIDをデコードして確認
for token_id in ids:
    print(f"{token_id} -> '{tokenizer.decode([token_id])}'")