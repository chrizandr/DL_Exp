"""
Vanilla PyTorch Implementation of 'Attention is All You Need'
https://arxiv.org/pdf/1706.03762

Author: chrizandr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadSDPAttentionBlock(nn.Module):
    def __init__(self, heads, dk, dv):
        super().__init__()
        self.attention_blocks = [SDPAttentionBlock(dk, dv, heads) for i in range(heads)]
        self.WO = torch.nn.Linear(dv, dv, bias=False)

    def forward(self, keys, queries, values, masking=False):
        out = [x(keys, queries, values, masking) for x in self.attention_blocks]
        out = torch.concat(out, 2) # B x N x (D/h) -> B x N x D
        out = self.WO(out)
        return out


class SDPAttentionBlock(nn.Module):
    def __init__(self, dk, dv, heads=1):
        super().__init__()
        self.WQ = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WK = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WV = torch.nn.Linear(dv, int(dv / heads), bias=False)


    def forward(self, keys, queries, values, masking=False):
        Q_ = self.WQ(queries)
        K_ = self.WK(keys)
        K_ = K_.permute(0, 2, 1)

        y = torch.bmm(Q_, K_) / np.sqrt(keys.shape[1])
        if masking:
            key_shape = keys.shape[1]
            queries_shape = queries.shape[1]
            mask = torch.tensor([[0 if i <= j else -1 * float("inf")
                                for i in range(key_shape)] for j in range(queries_shape)])
            y = mask + y

        z = F.softmax(y, 1)
        output = torch.matmul(z, self.WV(values))
        return output


class NNBlock(nn.Module):
    def __init__(self, dv, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(dv, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, dv, bias=True)

    def forward(self, values):
        z1 = F.relu(self.l1(values))
        z2 = self.l2(z1)
        return z2


def positional_encodings(dk, context_length=2048):
    i = np.array(list(range(dk)))
    even_mask = (i % 2 == 0).astype(int)
    encodings = []
    power = 10000 ** (2 * i / dk)

    for pos in range(context_length):
        angles = pos / power
        sins = np.sin(angles)
        coses = np.cos(angles)
        encoding = sins * even_mask + coses * (1-even_mask)
        encodings.append(encoding)
    return torch.tensor(np.array(encodings)).float()


class EncoderBlock(nn.Module):
    def __init__(self, dk, dv, hidden_dim, heads):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(dv)
        self.lnorm2 = nn.LayerNorm(dv)
        self.attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.ffn = NNBlock(dv, hidden_dim)

    def forward(self, keys, queries, values):
        z = self.attention(keys, queries, values)
        z = self.lnorm1(z + queries)
        y = self.ffn(z)
        y = self.lnorm2(y + z)
        return y


class Encoder(nn.Module):
    def __init__(self, num_blocks, dk, dv, hidden_dim, heads):
        super().__init__()
        self.blocks = [EncoderBlock(dk, dv, hidden_dim, heads) for _ in range(num_blocks)]

    def forward(self, keys, queries, values):
        for block in self.blocks:
            out = block(keys, queries, values)
            keys, queries, values = out, out, out
        return values


class DecoderBlock(nn.Module):
    def __init__(self, dk, dv, hidden_dim, heads):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(dv)
        self.lnorm2 = nn.LayerNorm(dv)
        self.lnorm3 = nn.LayerNorm(dv)
        self.attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.cross_attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.ffn = NNBlock(dv, hidden_dim)

    def forward(self, keys, queries, values, encoder_values):
        z = self.attention(keys, queries, values, masking=True)
        z = self.lnorm1(z + queries)
        y = self.cross_attention(encoder_values, z, encoder_values)
        y = self.lnorm2(y + z)
        out = self.ffn(y)
        out = self.lnorm3(y + out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_blocks, dk, dv, hidden_dim, heads, vocab_size):
        super().__init__()
        self.blocks = [
            DecoderBlock(dk, dv, hidden_dim, heads)
            for _ in range(num_blocks)
        ]
        self.output_layer = nn.Linear(dv, vocab_size)

    def forward(self, keys, queries, values, encoder_values):
        for block in self.blocks:
            out = block(keys, queries, values, encoder_values)
            keys, queries, values = out, out, out
        logits = self.output_layer(values)
        return logits


class Transformer(nn.Module):
    def __init__(self, context_length, dmodel, vocab_size, hidden_dim, heads, num_blocks=4):
        super().__init__()
        self.context_length = context_length
        self.pos = positional_encodings(dmodel, context_length)
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.encoder = Encoder(num_blocks, dmodel, dmodel, hidden_dim, heads)
        self.decoder = Decoder(num_blocks, dmodel, dmodel, hidden_dim, heads, vocab_size)

    def forward(self, input_sequence, output_sequence):
        assert len(input_sequence) < self.context_length
        assert len(output_sequence) < self.context_length
        input_embeddings = self.embedding(input_sequence)
        input_embeddings = input_embeddings + self.pos[0:input_embeddings.shape[1]]   # batching is forced
        encoded = self.encoder(input_embeddings, input_embeddings, input_embeddings)

        output_embeddings = self.embedding(output_sequence)
        output_embeddings = output_embeddings + self.pos[0:output_embeddings.shape[1]] # batching is forced
        decoded = self.decoder(output_embeddings, output_embeddings, output_embeddings, encoded)

        return decoded[:, -1, ::]   # B x N x D --> B x D (last element in each sequence of batch)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    dmodel = 128
    heads = 4
    hidden_dim = 2048
    vocab_size = 10000
    context_length = 512

    model = Transformer(context_length, dmodel, vocab_size, hidden_dim, heads)
    out = model(torch.tensor([[10, 1223, 23, 345, 234], [10, 1223, 23, 345, 234]]),
                torch.tensor([[10, 1223, 23, 345, 234, 2323, 232, 2321, 2313, 12], [10, 1223, 23, 345, 234, 2323, 232, 2321, 2313, 12]]))
    breakpoint()
