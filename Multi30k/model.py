from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from utils import DEVICE


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 # 指定了一个远大于输入句子长度的数值作为max_len
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # torch.arange(0, emb_size, 2): 从0-emb_size，逐项+2
        # math.log以e为底
        # den = e^(2i*ln^{10000}/emb_size) = e^{ln^{10000^{2i/emb_size}}} = 10000^{2i/emb_size}
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 初始化一个长度为max_len的1维数组，并转换为max_len*1的数组
        # [[0],[1],...[max_len-1]]
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 从第0个元素起，步长为2，对dim中第2i项赋值, *用来对tensor进行矩阵行逐元素相乘：
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # 从第1个元素起，步长为2赋值，对dim中第2i+1项赋值
        # [5000, 512] [max_len, dim]
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # [5000, 1, 512] [max_len, batch_size, dim] 为了与后续词嵌入匹配规格来相加
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        # 后续加载模型参数时，用于直接载入该部分模型的常数，即位置编码，而无需重复计算
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # [27, 128] [padded_length, batch_size]
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # print("src: ", src.size())
        # [27, 128], [padded_length, batch_size]
        # print("trg: ", trg.size())
        # [23, 128], [padded_length, batch_size]
        # print("src_mask: ", src_mask.size())
        # [27, 27], [padded_length, padded_length]
        # print("src_mask: ", src_mask)
        # print("tgt_mask: ", tgt_mask.size())
        # [23, 23], [padded_length, padded_length]
        # print("tgt_mask: ", tgt_mask)
        # print("src_padding_mask: ", src_padding_mask.size())
        # [128, 27] [batch_size, padded_length]
        # print("tgt_padding_mask: ", tgt_padding_mask.size())
        # [128, 23] [batch_size, padded_length]
        # print("memory_key_padding_mask: ", memory_key_padding_mask.size())
        # [128, 27] [batch_size, padded_length]
        src = self.src_tok_emb(src)
        # print("src: ", src.size())
        # [27, 128, 512] [padded_length, batch_size, dim]
        src_emb = self.positional_encoding(src)
        # print("src_embed: ", src_emb.size())
        # [27, 128, 512] [padded_length, batch_size, dim]
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

        # [23, 128, 10837] [padded_length, batch_size, vocab_size]
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)